"""Diagnostic REML/profile-MAP refinements for Gaussian GLMM variance scales.

This is an experiment-only script. It compares the production marginal MAP
baseline against narrower variance-component refinements initialized from the
MoM/EM estimator. The production estimator is left untouched.

Examples:
    uv run python experiments/analytical/glmm_reml_diagnostic.py
    uv run python experiments/analytical/glmm_reml_diagnostic.py --max-batches 1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.glmm import glmm
from metabeta.analytical.map import _fixedCorrFromStats, _logMarginalTarget, _replacePsiDiag
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import datasetFilename


SIZES = ['small', 'medium', 'large', 'huge']


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    data_dir = ROOT / 'metabeta' / 'outputs' / 'data' / cfg['data_id']
    if partition == 'train':
        return [data_dir / datasetFilename('train', ep) for ep in range(1, n_epochs + 1)]
    return [data_dir / f'{partition}.npz']


def _refine_variance_scales(
    center: dict[str, torch.Tensor],
    fallback: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
    optimize_sigma_eps: bool,
    n_steps: int,
    lr: float,
) -> dict[str, torch.Tensor]:
    """Refine sigma(RFX), optionally sigma(Eps), with beta/correlation fixed."""
    if max_q == 0 or n_steps <= 0:
        return fallback

    corr = _fixedCorrFromStats(
        center,
        batch.get('eta_rfx'),
        batch.get('mask_q'),
        max_q,
    ).detach()
    beta = center['beta_est'].detach()
    log_sigma_rfx = (
        center['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    log_sigma_eps = (
        center['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log().clone()
    )
    params: list[torch.Tensor] = [log_sigma_rfx]
    if optimize_sigma_eps:
        log_sigma_eps = log_sigma_eps.clone().requires_grad_(True)
        params.append(log_sigma_eps)

    optimizer = torch.optim.Adam(params, lr=lr)
    valid_rows = torch.ones(beta.shape[0], dtype=torch.bool, device=beta.device)
    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            target = _logMarginalTarget(
                beta.unsqueeze(1),
                log_sigma_rfx.unsqueeze(1),
                log_sigma_eps.unsqueeze(1),
                corr,
                batch['X'][..., : beta.shape[-1]],
                batch['y'],
                batch['Z'][..., :max_q],
                batch['mask_n'].float(),
                batch['mask_m'].float(),
                batch['nu_ffx'],
                batch['tau_ffx'],
                batch['family_ffx'],
                batch['tau_rfx'],
                batch['family_sigma_rfx'],
                batch['tau_eps'],
                batch['family_sigma_eps'],
                batch.get('mask_d'),
                batch.get('mask_q'),
            ).squeeze(1)
            finite_target = torch.isfinite(target)
            active_rows = valid_rows & finite_target
            valid_rows = valid_rows & finite_target
            if not bool(active_rows.any()):
                break
            loss = -target[active_rows].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()
            with torch.no_grad():
                log_sigma_rfx.clamp_(math.log(1e-4), math.log(20.0))
                if optimize_sigma_eps:
                    log_sigma_eps.clamp_(math.log(1e-4), math.log(20.0))

    sigma_rfx = log_sigma_rfx.detach().exp()
    sigma_eps = log_sigma_eps.detach().exp()
    valid = valid_rows & torch.isfinite(sigma_rfx).all(dim=-1)
    valid = valid & (sigma_rfx >= 1e-4).all(dim=-1) & (sigma_rfx <= 20.0).all(dim=-1)
    if optimize_sigma_eps:
        valid = valid & torch.isfinite(sigma_eps) & (sigma_eps >= 1e-4) & (sigma_eps <= 20.0)

    if 'mask_q' in batch:
        mask_q = batch['mask_q'][..., :max_q].bool()
        sigma_rfx = torch.where(mask_q, sigma_rfx, fallback['sigma_rfx_est'][..., :max_q])
    sigma_rfx = torch.where(valid[:, None], sigma_rfx, fallback['sigma_rfx_est'][..., :max_q])

    out = dict(fallback)
    out['sigma_rfx_est'] = sigma_rfx
    if optimize_sigma_eps:
        sigma_eps = torch.where(valid, sigma_eps, fallback['sigma_eps_est'].squeeze(-1))
        out['sigma_eps_est'] = sigma_eps.unsqueeze(-1)
    if 'Psi' in fallback:
        out['Psi'] = _replacePsiDiag(fallback['Psi'], sigma_rfx, batch.get('mask_q'))
    return out


class _MetricStore:
    def __init__(self) -> None:
        self.beta_errs: list[np.ndarray] = []
        self.beta_truths: list[np.ndarray] = []
        self.srfx_errs: list[np.ndarray] = []
        self.srfx_truths: list[np.ndarray] = []
        self.seps_errs: list[np.ndarray] = []
        self.seps_truths: list[np.ndarray] = []
        self.blup_errs: list[np.ndarray] = []
        self.blup_truths: list[np.ndarray] = []
        self.seconds = 0.0

    def add(self, est: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], max_q: int) -> None:
        mask_d = batch['mask_d'].bool()
        mask_q = batch['mask_q'][..., :max_q].bool()
        mask_m = batch['mask_m'].bool()
        B = batch['X'].shape[0]
        for b in range(B):
            self.beta_errs.append(
                (est['beta_est'][b][mask_d[b]] - batch['ffx'][b][mask_d[b]]).cpu().numpy()
            )
            self.beta_truths.append(batch['ffx'][b][mask_d[b]].cpu().numpy())
            self.srfx_errs.append(
                (est['sigma_rfx_est'][b][mask_q[b]] - batch['sigma_rfx'][b][mask_q[b]])
                .cpu()
                .numpy()
            )
            self.srfx_truths.append(batch['sigma_rfx'][b][mask_q[b]].cpu().numpy())
            self.seps_errs.append(
                (est['sigma_eps_est'][b, 0] - batch['sigma_eps'][b]).reshape(1).cpu().numpy()
            )
            self.seps_truths.append(batch['sigma_eps'][b].reshape(1).cpu().numpy())
            blup_est = est['blup_est'][b][mask_m[b]][:, mask_q[b]]
            blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
            self.blup_errs.append((blup_est - blup_true).reshape(-1).cpu().numpy())
            self.blup_truths.append(blup_true.reshape(-1).cpu().numpy())

    def row(self, method: str, dataset: str, partition: str, n_total: int) -> str:
        values = [
            method,
            dataset,
            partition,
            str(n_total),
            f'{_nrmse(np.concatenate(self.beta_errs), np.concatenate(self.beta_truths)):.4f}',
            f'{_nrmse(np.concatenate(self.srfx_errs), np.concatenate(self.srfx_truths)):.4f}',
            f'{_nrmse(np.concatenate(self.seps_errs), np.concatenate(self.seps_truths)):.4f}',
            f'{_nrmse(np.concatenate(self.blup_errs), np.concatenate(self.blup_truths)):.4f}',
            f'{self.seconds:.2f}',
        ]
        return ','.join(values)


def run(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(True)
    device = torch.device(args.device)

    print('method,dataset,partition,N,FFX,sRFX,sEps,BLUP,seconds')
    for size in args.sizes:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            stores: dict[str, _MetricStore] = defaultdict(_MetricStore)
            n_total = 0

            for path in _paths(data_id, partition, n_epochs):
                for batch_idx, batch in enumerate(
                    Dataloader(path, batch_size=args.batch_size, shuffle=False)
                ):
                    if args.max_batches is not None and batch_idx >= args.max_batches:
                        break
                    batch = toDevice(batch, device)
                    Zm = batch['Z'][..., :max_q]
                    common = dict(
                        eta_rfx=batch.get('eta_rfx'),
                        mask_q=batch.get('mask_q'),
                        nu_ffx=batch.get('nu_ffx'),
                        tau_ffx=batch.get('tau_ffx'),
                        family_ffx=batch.get('family_ffx'),
                        tau_rfx=batch.get('tau_rfx'),
                        family_sigma_rfx=batch.get('family_sigma_rfx'),
                        tau_eps=batch.get('tau_eps'),
                        family_sigma_eps=batch.get('family_sigma_eps'),
                        mask_d=batch.get('mask_d'),
                    )

                    start = time.perf_counter()
                    with torch.no_grad():
                        current = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            **common,
                        )
                    current_seconds = time.perf_counter() - start
                    stores['current'].seconds += current_seconds
                    stores['current'].add(current, batch, max_q)

                    start = time.perf_counter()
                    with torch.no_grad():
                        mom_em = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            map_refine=False,
                            **common,
                        )
                    mom_seconds = time.perf_counter() - start
                    stores['mom_em'].seconds += mom_seconds
                    stores['mom_em'].add(mom_em, batch, max_q)

                    start = time.perf_counter()
                    reml_diag = _refine_variance_scales(
                        mom_em,
                        current,
                        batch,
                        max_q,
                        optimize_sigma_eps=False,
                        n_steps=args.n_steps,
                        lr=args.lr,
                    )
                    stores['reml_diag'].seconds += time.perf_counter() - start
                    stores['reml_diag'].add(reml_diag, batch, max_q)

                    start = time.perf_counter()
                    reml_diag_seps = _refine_variance_scales(
                        mom_em,
                        current,
                        batch,
                        max_q,
                        optimize_sigma_eps=True,
                        n_steps=args.n_steps,
                        lr=args.lr,
                    )
                    stores['reml_diag_seps'].seconds += time.perf_counter() - start
                    stores['reml_diag_seps'].add(reml_diag_seps, batch, max_q)

                    n_total += batch['X'].shape[0]

            for method in ['current', 'mom_em', 'reml_diag', 'reml_diag_seps']:
                print(stores[method].row(method, data_id, partition, n_total))


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=SIZES, choices=SIZES)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.03)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run(setup())
