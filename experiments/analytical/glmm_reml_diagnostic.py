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
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.glmm import glmm
from metabeta.analytical.reml import gateNormalRemlVsMap, refineNormalRemlSrfx
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


class _BreakdownBucket:
    def __init__(self) -> None:
        self.mom_errs: list[np.ndarray] = []
        self.current_errs: list[np.ndarray] = []
        self.reml_errs: list[np.ndarray] = []
        self.gated_errs: list[np.ndarray] = []
        self.truths: list[np.ndarray] = []
        self.n_rows = 0
        self.valid_rows = 0
        self.clamped_rows = 0
        self.gated_rows = 0
        self.both_worse_rows = 0

    def add(
        self,
        mom_err: np.ndarray,
        current_err: np.ndarray,
        reml_err: np.ndarray,
        gated_err: np.ndarray,
        truth: np.ndarray,
        valid: bool,
        clamped: bool,
        gated: bool,
    ) -> None:
        self.mom_errs.append(mom_err)
        self.current_errs.append(current_err)
        self.reml_errs.append(reml_err)
        self.gated_errs.append(gated_err)
        self.truths.append(truth)
        self.n_rows += 1
        self.valid_rows += int(valid)
        self.clamped_rows += int(clamped)
        self.gated_rows += int(gated)
        mom_sse = float(np.square(mom_err).sum())
        current_sse = float(np.square(current_err).sum())
        reml_sse = float(np.square(reml_err).sum())
        self.both_worse_rows += int(current_sse > mom_sse and reml_sse > mom_sse)

    def row(self, label: str) -> str:
        mom = _nrmse(np.concatenate(self.mom_errs), np.concatenate(self.truths))
        current = _nrmse(np.concatenate(self.current_errs), np.concatenate(self.truths))
        reml = _nrmse(np.concatenate(self.reml_errs), np.concatenate(self.truths))
        gated = _nrmse(np.concatenate(self.gated_errs), np.concatenate(self.truths))
        values = [
            label,
            str(self.n_rows),
            f'{mom:.4f}',
            f'{current:.4f}',
            f'{reml:.4f}',
            f'{gated:.4f}',
            f'{reml - current:.4f}',
            f'{100.0 * (current - reml) / max(current, 1e-8):.2f}',
            f'{1.0 - self.valid_rows / max(self.n_rows, 1):.4f}',
            f'{self.clamped_rows / max(self.n_rows, 1):.4f}',
            f'{self.gated_rows / max(self.n_rows, 1):.4f}',
            f'{self.both_worse_rows / max(self.n_rows, 1):.4f}',
        ]
        return ','.join(values)


class _BreakdownStore:
    def __init__(self) -> None:
        self.buckets: dict[str, _BreakdownBucket] = defaultdict(_BreakdownBucket)

    def add_batch(
        self,
        current: dict[str, torch.Tensor],
        mom_em: dict[str, torch.Tensor],
        reml: dict[str, torch.Tensor],
        gated: dict[str, torch.Tensor],
        use_reml_gate: torch.Tensor,
        meta,
        batch: dict[str, torch.Tensor],
        dataset: str,
        partition: str,
        max_q: int,
    ) -> None:
        mask_d = batch['mask_d'].bool()
        mask_q = batch['mask_q'][..., :max_q].bool()
        qs = mask_q.sum(dim=-1).cpu().numpy()
        ds = mask_d.sum(dim=-1).cpu().numpy()
        ms = batch['mask_m'].sum(dim=-1).cpu().numpy()
        ns = batch['n'].cpu().numpy()
        eta = batch.get('eta_rfx')
        if eta is None:
            etas = np.ones_like(qs)
        else:
            etas = eta.cpu().numpy()

        fallback = ~meta.valid.cpu().numpy()
        clamped = meta.clamped.cpu().numpy()
        B = batch['X'].shape[0]
        for b in range(B):
            q_mask = mask_q[b]
            truth = batch['sigma_rfx'][b][q_mask].cpu().numpy()
            mom_err = mom_em['sigma_rfx_est'][b][q_mask] - batch['sigma_rfx'][b][q_mask]
            current_err = current['sigma_rfx_est'][b][q_mask] - batch['sigma_rfx'][b][q_mask]
            reml_err = reml['sigma_rfx_est'][b][q_mask] - batch['sigma_rfx'][b][q_mask]
            gated_err = gated['sigma_rfx_est'][b][q_mask] - batch['sigma_rfx'][b][q_mask]
            mom_err_np = mom_err.cpu().numpy()
            current_err_np = current_err.cpu().numpy()
            reml_err_np = reml_err.cpu().numpy()
            gated_err_np = gated_err.cpu().numpy()
            mom_sigma = mom_em['sigma_rfx_est'][b][q_mask]
            current_sigma = current['sigma_rfx_est'][b][q_mask]
            map_rel_delta = (
                (current_sigma - mom_sigma).abs() / mom_sigma.abs().clamp(min=1e-8)
            ).amax()
            map_direction = (current_sigma.mean() - mom_sigma.mean()).item()
            labels = [
                'all',
                f'dataset={dataset}/{partition}',
                f'partition={partition}',
                f'q={_q_bin(int(qs[b]))}',
                f'd={_d_bin(int(ds[b]))}',
                f'm={_m_bin(float(ms[b]))}',
                f'n={_n_bin(float(ns[b]))}',
                f'eta_rfx={int(etas[b] > 0)}',
                f'map_delta={_map_delta_bin(float(map_rel_delta))}',
                f'map_direction={_map_direction_bin(map_direction)}',
                f'true_sigma={_sigma_bin(float(np.mean(truth)))}',
            ]
            for label in labels:
                self.buckets[label].add(
                    mom_err_np,
                    current_err_np,
                    reml_err_np,
                    gated_err_np,
                    truth,
                    valid=not bool(fallback[b]),
                    clamped=bool(clamped[b]),
                    gated=bool(use_reml_gate[b]),
                )

    def print_rows(self) -> None:
        print('')
        print(
            'breakdown,N,mom_em_sRFX,current_sRFX,reml_diag_sRFX,reml_gated_sRFX,delta,'
            'rel_improve_pct,fallback_rate,clamp_rate,gate_rate,both_worse_than_mom_rate'
        )
        ordered = ['all']
        ordered.extend(sorted(label for label in self.buckets if label != 'all'))
        for label in ordered:
            print(self.buckets[label].row(label))


def _q_bin(q: int) -> str:
    if q <= 1:
        return '1'
    if q == 2:
        return '2'
    return '3+'


def _d_bin(d: int) -> str:
    if d <= 4:
        return '<=4'
    if d <= 8:
        return '5-8'
    return '9+'


def _m_bin(m: float) -> str:
    if m < 20:
        return '<20'
    if m < 50:
        return '20-49'
    return '50+'


def _n_bin(n: float) -> str:
    if n < 500:
        return '<500'
    if n < 2000:
        return '500-1999'
    return '2000+'


def _map_delta_bin(delta: float) -> str:
    if delta < 1e-3:
        return '<0.1pct'
    if delta < 0.05:
        return '0.1-5pct'
    if delta < 0.20:
        return '5-20pct'
    return '20pct+'


def _map_direction_bin(delta: float) -> str:
    if abs(delta) < 1e-6:
        return 'none'
    if delta < 0:
        return 'shrink'
    return 'expand'


def _sigma_bin(sigma: float) -> str:
    if sigma < 0.25:
        return '<0.25'
    if sigma < 0.75:
        return '0.25-0.75'
    if sigma < 1.5:
        return '0.75-1.5'
    return '1.5+'


def run(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(True)
    device = torch.device(args.device)
    breakdown = _BreakdownStore() if args.breakdown else None

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
                    reml_diag, reml_meta = refineNormalRemlSrfx(
                        mom_em,
                        current,
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['nu_ffx'],
                        batch['tau_ffx'],
                        batch['family_ffx'],
                        batch['tau_rfx'],
                        batch['family_sigma_rfx'],
                        batch['tau_eps'],
                        batch['family_sigma_eps'],
                        eta_rfx=batch.get('eta_rfx'),
                        mask_d=batch.get('mask_d'),
                        mask_q=batch.get('mask_q'),
                        optimize_sigma_eps=False,
                        n_steps=args.n_steps,
                        lr=args.lr,
                    )
                    stores['reml_diag'].seconds += time.perf_counter() - start
                    stores['reml_diag'].add(reml_diag, batch, max_q)

                    gated, use_reml_gate = gateNormalRemlVsMap(
                        current,
                        reml_diag,
                        reml_meta,
                        batch['n'],
                        mask_q=batch.get('mask_q'),
                        min_q=args.gate_min_q,
                        max_n_total=args.gate_max_n,
                    )
                    stores['reml_gated'].add(gated, batch, max_q)
                    if breakdown is not None:
                        breakdown.add_batch(
                            current,
                            mom_em,
                            reml_diag,
                            gated,
                            use_reml_gate,
                            reml_meta,
                            batch,
                            data_id,
                            partition,
                            max_q,
                        )

                    start = time.perf_counter()
                    reml_diag_seps, _ = refineNormalRemlSrfx(
                        mom_em,
                        current,
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['nu_ffx'],
                        batch['tau_ffx'],
                        batch['family_ffx'],
                        batch['tau_rfx'],
                        batch['family_sigma_rfx'],
                        batch['tau_eps'],
                        batch['family_sigma_eps'],
                        eta_rfx=batch.get('eta_rfx'),
                        mask_d=batch.get('mask_d'),
                        mask_q=batch.get('mask_q'),
                        optimize_sigma_eps=True,
                        n_steps=args.n_steps,
                        lr=args.lr,
                    )
                    stores['reml_diag_seps'].seconds += time.perf_counter() - start
                    stores['reml_diag_seps'].add(reml_diag_seps, batch, max_q)

                    n_total += batch['X'].shape[0]

            stores['reml_gated'].seconds = stores['reml_diag'].seconds
            for method in ['current', 'mom_em', 'reml_diag', 'reml_gated', 'reml_diag_seps']:
                print(stores[method].row(method, data_id, partition, n_total))

    if breakdown is not None:
        breakdown.print_rows()


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=SIZES, choices=SIZES)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--gate-min-q', type=int, default=2)
    parser.add_argument('--gate-max-n', type=int, default=1999)
    parser.add_argument('--breakdown', action='store_true')
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run(setup())
