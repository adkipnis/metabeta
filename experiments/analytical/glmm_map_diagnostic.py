"""Compare MoM/EM GLMM estimates with short MAP hybrid alternatives.

This is an experiment-only diagnostic. It leaves the production analytical
estimator untouched and asks whether prior-aware FFX/sRFX estimates can improve
global context while keeping MoM/EM sigma(Eps) and BLUP outputs.

Examples:
    uv run python experiments/analytical/glmm_map_diagnostic.py
    uv run python experiments/analytical/glmm_map_diagnostic.py --max-batches 1
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
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.families import logProbFfx, logProbSigma
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


def _active_mask(batch: dict[str, torch.Tensor], key: str, width: int) -> torch.Tensor:
    return batch[key][..., :width].bool()


def _fixed_corr_from_stats(
    stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    q: int,
) -> torch.Tensor:
    """Extract a stable fixed correlation matrix from current GLMM Psi output."""
    device = stats['Psi'].device
    dtype = stats['Psi'].dtype
    B = stats['Psi'].shape[0]
    eye = torch.eye(q, device=device, dtype=dtype).expand(B, q, q)

    Psi = stats['Psi'][..., :q, :q]
    std = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    corr = Psi / (std[:, :, None] * std[:, None, :])
    corr = corr.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.95, 0.95)
    corr = 0.5 * (corr + corr.mT)
    corr = corr - torch.diag_embed(corr.diagonal(dim1=-2, dim2=-1)) + eye

    if 'eta_rfx' in batch:
        corr_active = batch['eta_rfx'].to(device=device) > 0
        corr = torch.where(corr_active[:, None, None], corr, eye)

    if 'mask_q' in batch:
        mask_q = _active_mask(batch, 'mask_q', q).to(device=device)
        active_qq = mask_q[:, :, None] & mask_q[:, None, :]
        corr = torch.where(active_qq, corr, eye)

    vals, vecs = torch.linalg.eigh(corr)
    corr = vecs @ torch.diag_embed(vals.clamp(min=1e-4)) @ vecs.mT
    corr_std = corr.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    corr = corr / (corr_std[:, :, None] * corr_std[:, None, :])
    return 0.5 * (corr + corr.mT)


def _psi_from_sigma_corr(sigma_rfx: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    return corr[:, None, :, :] * sigma_rfx[:, :, :, None] * sigma_rfx[:, :, None, :]


def _corr_cholesky(corr: torch.Tensor) -> torch.Tensor:
    q = corr.shape[-1]
    eye = torch.eye(q, device=corr.device, dtype=corr.dtype)
    return torch.linalg.cholesky(corr + 1e-6 * eye)


def _rfx_from_u(
    sigma_rfx: torch.Tensor,  # (B, S, q)
    corr: torch.Tensor,  # (B, q, q)
    u: torch.Tensor,  # (B, m, S, q)
) -> torch.Tensor:
    L_corr = _corr_cholesky(corr)
    L_full = sigma_rfx.unsqueeze(-1) * L_corr[:, None, :, :]
    return torch.einsum('bmsi,bsji->bmsj', u, L_full)


def _u_from_rfx(
    sigma_rfx: torch.Tensor,  # (B, S, q)
    corr: torch.Tensor,  # (B, q, q)
    rfx: torch.Tensor,  # (B, m, S, q)
) -> torch.Tensor:
    B, m, S, q = rfx.shape
    L_corr = _corr_cholesky(corr)
    L_full = sigma_rfx.unsqueeze(-1) * L_corr[:, None, :, :]
    L_exp = L_full[:, None].expand(B, m, S, q, q)
    return torch.linalg.solve_triangular(L_exp, rfx.unsqueeze(-1), upper=False).squeeze(-1)


def _log_marginal_likelihood_full(
    beta: torch.Tensor,  # (B, S, d)
    sigma_rfx: torch.Tensor,  # (B, S, q)
    sigma_eps: torch.Tensor,  # (B, S)
    corr: torch.Tensor,  # (B, q, q)
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normal marginal log-likelihood with a fixed random-effect correlation."""
    X = batch['X'][..., : beta.shape[-1]]
    Z = batch['Z'][..., : sigma_rfx.shape[-1]]
    y = batch['y'].unsqueeze(-1)
    mask_n = batch['mask_n'].unsqueeze(-1)
    mask_m = batch['mask_m']

    B, S, q = sigma_rfx.shape
    Psi = _psi_from_sigma_corr(sigma_rfx.clamp(min=1e-6), corr)
    eye_q = torch.eye(q, device=Psi.device, dtype=Psi.dtype)
    jitter = 1e-6 * eye_q

    chol_Psi, info_Psi = torch.linalg.cholesky_ex(Psi + jitter)
    psi_ok = info_Psi == 0
    eye_bs = eye_q.expand(B, S, q, q)
    Psi_inv = torch.cholesky_solve(eye_bs, chol_Psi)
    log_det_Psi = 2.0 * chol_Psi.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    Z_m = Z * mask_n
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)

    mu = torch.einsum('bmnd,bsd->bmns', X, beta)
    r = (y - mu) * mask_n
    rtr = r.square().sum(dim=2)
    h = torch.einsum('bmnq,bmns->bmsq', Z_m, r)
    n_i = mask_n.squeeze(-1).sum(dim=-1).float()

    s2e = sigma_eps.clamp(min=1e-6).square()
    M = Psi_inv[:, None] + ZtZ[:, :, None] / s2e[:, None, :, None, None]
    chol_M, info_M = torch.linalg.cholesky_ex(M + jitter)
    M_ok = info_M == 0

    log_det_M = 2.0 * chol_M.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    M_inv_h = torch.cholesky_solve(h.unsqueeze(-1), chol_M).squeeze(-1)
    hMh = (h * M_inv_h).sum(-1)

    log_det_V = log_det_M + log_det_Psi[:, None, :] + n_i[:, :, None] * s2e.log()[:, None, :]
    quad = (rtr - hMh / s2e[:, None, :]) / s2e[:, None, :]
    ll_i = -0.5 * (n_i[:, :, None] * math.log(2.0 * math.pi) + log_det_V + quad)

    active_m = mask_m.bool()[:, :, None]
    ok = M_ok & psi_ok[:, None, :]
    ll_i = torch.where(ok | ~active_m, ll_i, ll_i.new_tensor(-torch.inf))
    return torch.where(active_m, ll_i, ll_i.new_zeros(())).sum(dim=1)


def _log_target(
    beta: torch.Tensor,
    log_sigma_rfx: torch.Tensor,
    log_sigma_eps: torch.Tensor,
    corr: torch.Tensor,
    batch: dict[str, torch.Tensor],
    include_log_sigma_jacobian: bool,
) -> torch.Tensor:
    sigma_rfx = log_sigma_rfx.exp()
    sigma_eps = log_sigma_eps.exp()

    ll = _log_marginal_likelihood_full(beta, sigma_rfx, sigma_eps, corr, batch)
    mask_d = batch['mask_d'][..., : beta.shape[-1]].unsqueeze(-2).to(beta.dtype)
    mask_q = batch['mask_q'][..., : sigma_rfx.shape[-1]].unsqueeze(-2).to(beta.dtype)

    lp = logProbFfx(
        beta,
        batch['nu_ffx'][..., : beta.shape[-1]].unsqueeze(-2),
        batch['tau_ffx'][..., : beta.shape[-1]].unsqueeze(-2) + 1e-12,
        batch['family_ffx'],
        mask_d,
    )
    lp = lp + logProbSigma(
        sigma_rfx,
        batch['tau_rfx'][..., : sigma_rfx.shape[-1]].unsqueeze(-2) + 1e-12,
        batch['family_sigma_rfx'],
        mask_q,
    )
    lp = lp + logProbSigma(
        sigma_eps,
        batch['tau_eps'].unsqueeze(-1) + 1e-12,
        batch['family_sigma_eps'],
    )

    if include_log_sigma_jacobian:
        lp = lp + (log_sigma_rfx * mask_q).sum(-1) + log_sigma_eps
    return ll + lp


def _log_conditional_likelihood_normal(
    beta: torch.Tensor,  # (B, S, d)
    sigma_eps: torch.Tensor,  # (B, S)
    rfx: torch.Tensor,  # (B, m, S, q)
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    X = batch['X'][..., : beta.shape[-1]]
    Z = batch['Z'][..., : rfx.shape[-1]]
    y = batch['y'].unsqueeze(-1)
    mask_n = batch['mask_n'].unsqueeze(-1)
    mu_fixed = torch.einsum('bmnd,bsd->bmns', X, beta)
    mu_rfx = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    resid = (y - mu_fixed - mu_rfx) / sigma_eps[:, None, None, :].clamp(min=1e-6)
    ll = -0.5 * resid.square() - sigma_eps[:, None, None, :].clamp(min=1e-6).log()
    ll = ll - 0.5 * math.log(2.0 * math.pi)
    return (ll * mask_n).sum(dim=(1, 2))


def _log_full_target(
    beta: torch.Tensor,
    log_sigma_rfx: torch.Tensor,
    log_sigma_eps: torch.Tensor,
    u: torch.Tensor,
    corr: torch.Tensor,
    batch: dict[str, torch.Tensor],
    include_log_sigma_jacobian: bool,
) -> torch.Tensor:
    sigma_rfx = log_sigma_rfx.exp()
    sigma_eps = log_sigma_eps.exp()
    rfx = _rfx_from_u(sigma_rfx, corr, u)
    ll = _log_conditional_likelihood_normal(beta, sigma_eps, rfx, batch)

    mask_d = batch['mask_d'][..., : beta.shape[-1]].unsqueeze(-2).to(beta.dtype)
    mask_q = batch['mask_q'][..., : sigma_rfx.shape[-1]].unsqueeze(-2).to(beta.dtype)
    lp = logProbFfx(
        beta,
        batch['nu_ffx'][..., : beta.shape[-1]].unsqueeze(-2),
        batch['tau_ffx'][..., : beta.shape[-1]].unsqueeze(-2) + 1e-12,
        batch['family_ffx'],
        mask_d,
    )
    lp = lp + logProbSigma(
        sigma_rfx,
        batch['tau_rfx'][..., : sigma_rfx.shape[-1]].unsqueeze(-2) + 1e-12,
        batch['family_sigma_rfx'],
        mask_q,
    )
    lp = lp + logProbSigma(
        sigma_eps,
        batch['tau_eps'].unsqueeze(-1) + 1e-12,
        batch['family_sigma_eps'],
    )

    mask_u = batch['mask_m'][:, :, None, None].to(u.dtype) * batch['mask_q'][
        ..., : sigma_rfx.shape[-1]
    ][:, None, None, :].to(u.dtype)
    lp_u = (-0.5 * (u.square() + math.log(2.0 * math.pi)) * mask_u).sum(dim=(1, 3))
    lp = lp + lp_u

    if include_log_sigma_jacobian:
        lp = lp + (log_sigma_rfx * mask_q).sum(-1) + log_sigma_eps
    return ll + lp


def _pack_globals(
    beta: torch.Tensor,
    log_sigma_rfx: torch.Tensor,
    log_sigma_eps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        'beta_est': beta.detach(),
        'sigma_rfx_est': log_sigma_rfx.detach().exp(),
        'sigma_eps_est': log_sigma_eps.detach().exp().unsqueeze(-1),
    }


def _hybrid_estimate(
    stats: dict[str, torch.Tensor],
    est: dict[str, torch.Tensor],
    replace_beta: bool,
    replace_sigma_rfx: bool,
) -> dict[str, torch.Tensor]:
    out = {
        'beta_est': stats['beta_est'],
        'sigma_rfx_est': stats['sigma_rfx_est'],
        'sigma_eps_est': stats['sigma_eps_est'],
        'blup_est': stats['blup_est'],
    }
    if replace_beta:
        out['beta_est'] = est['beta_est']
    if replace_sigma_rfx:
        out['sigma_rfx_est'] = est['sigma_rfx_est']
    return out


def _run_map_marginal(
    center: dict[str, torch.Tensor],
    corr: torch.Tensor,
    batch: dict[str, torch.Tensor],
    checkpoints: list[int],
    lr: float,
) -> dict[str, dict[str, torch.Tensor]]:
    beta = center['beta_est'].detach().clone().requires_grad_(True)
    log_sigma_rfx = (
        center['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    log_sigma_eps = (
        center['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)

    optimizer = torch.optim.Adam([beta, log_sigma_rfx, log_sigma_eps], lr=lr)
    out: dict[str, dict[str, torch.Tensor]] = {}
    max_step = max(checkpoints)

    for step in range(1, max_step + 1):
        optimizer.zero_grad(set_to_none=True)
        target = _log_target(
            beta.unsqueeze(1),
            log_sigma_rfx.unsqueeze(1),
            log_sigma_eps.unsqueeze(1),
            corr,
            batch,
            include_log_sigma_jacobian=False,
        ).squeeze(1)
        loss = -target.sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta, log_sigma_rfx, log_sigma_eps], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-20.0, 20.0)
            log_sigma_rfx.clamp_(math.log(1e-4), math.log(20.0))
            log_sigma_eps.clamp_(math.log(1e-4), math.log(20.0))

        if step in checkpoints:
            out[f'map{step}'] = _pack_globals(beta, log_sigma_rfx, log_sigma_eps)

    for step in checkpoints:
        name = f'map{step}'
        if name not in out:
            out[name] = _pack_globals(beta, log_sigma_rfx, log_sigma_eps)
    return out


def _run_map_full(
    center: dict[str, torch.Tensor],
    corr: torch.Tensor,
    batch: dict[str, torch.Tensor],
    checkpoints: list[int],
    lr: float,
) -> dict[str, dict[str, torch.Tensor]]:
    beta = center['beta_est'].detach().clone().requires_grad_(True)
    log_sigma_rfx = (
        center['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    log_sigma_eps = (
        center['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    sigma_rfx = center['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).unsqueeze(1)
    rfx = center['blup_est'].detach().unsqueeze(2)
    u = _u_from_rfx(sigma_rfx, corr, rfx).squeeze(2).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([beta, log_sigma_rfx, log_sigma_eps, u], lr=lr)
    out: dict[str, dict[str, torch.Tensor]] = {}
    max_step = max(checkpoints)

    for step in range(1, max_step + 1):
        optimizer.zero_grad(set_to_none=True)
        target = _log_full_target(
            beta.unsqueeze(1),
            log_sigma_rfx.unsqueeze(1),
            log_sigma_eps.unsqueeze(1),
            u.unsqueeze(2),
            corr,
            batch,
            include_log_sigma_jacobian=False,
        ).squeeze(1)
        loss = -target.sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta, log_sigma_rfx, log_sigma_eps, u], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-20.0, 20.0)
            log_sigma_rfx.clamp_(math.log(1e-4), math.log(20.0))
            log_sigma_eps.clamp_(math.log(1e-4), math.log(20.0))
            u.clamp_(-20.0, 20.0)

        if step in checkpoints:
            out[f'map{step}'] = _pack_globals(beta, log_sigma_rfx, log_sigma_eps)

    for step in checkpoints:
        name = f'map{step}'
        if name not in out:
            out[name] = _pack_globals(beta, log_sigma_rfx, log_sigma_eps)
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

    def add(
        self,
        est: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        max_q: int,
    ) -> None:
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


def _parse_ints(value: str) -> list[int]:
    out = sorted({int(v.strip()) for v in value.split(',') if v.strip()})
    if not out:
        raise ValueError('expected at least one integer')
    return out


def run(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(True)
    device = torch.device(args.device)
    map_steps = _parse_ints(args.map_steps)

    print('method,dataset,partition,N,FFX,sRFX,sEps,BLUP,seconds')
    for size in args.sizes:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            stores: dict[str, _MetricStore] = defaultdict(_MetricStore)
            method_order: list[str] = []
            seen_methods: set[str] = set()
            n_total = 0

            def add_method(name: str) -> None:
                if name not in seen_methods:
                    seen_methods.add(name)
                    method_order.append(name)

            for path in _paths(data_id, partition, n_epochs):
                for batch_idx, batch in enumerate(
                    Dataloader(path, batch_size=args.batch_size, shuffle=False)
                ):
                    if args.max_batches is not None and batch_idx >= args.max_batches:
                        break
                    batch = toDevice(batch, device)
                    Zm = batch['Z'][..., :max_q]
                    baseline_start = time.perf_counter()
                    with torch.no_grad():
                        stats = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            eta_rfx=batch.get('eta_rfx'),
                            mask_q=batch.get('mask_q'),
                        )
                    stores['mom_em'].seconds += time.perf_counter() - baseline_start
                    stores['mom_em'].add(stats, batch, max_q)
                    add_method('mom_em')
                    n_total += batch['X'].shape[0]

                    corr = _fixed_corr_from_stats(stats, batch, max_q)

                    def record(
                        method: str,
                        est: dict[str, torch.Tensor],
                        seconds: float,
                    ) -> None:
                        ffx_method = f'{method}_hyb_ffx'
                        ffx_est = _hybrid_estimate(
                            stats, est, replace_beta=True, replace_sigma_rfx=False
                        )
                        stores[ffx_method].seconds += seconds
                        stores[ffx_method].add(ffx_est, batch, max_q)
                        add_method(ffx_method)

                        srfx_method = f'{method}_hyb_srfx'
                        srfx_est = _hybrid_estimate(
                            stats, est, replace_beta=False, replace_sigma_rfx=True
                        )
                        stores[srfx_method].seconds += seconds
                        stores[srfx_method].add(srfx_est, batch, max_q)
                        add_method(srfx_method)

                        srfx_method = f'{method}_hyb_ffx_srfx'
                        srfx_est = _hybrid_estimate(
                            stats, est, replace_beta=True, replace_sigma_rfx=True
                        )
                        stores[srfx_method].seconds += seconds
                        stores[srfx_method].add(srfx_est, batch, max_q)
                        add_method(srfx_method)

                    map_start = time.perf_counter()
                    map_estimates = _run_map_marginal(stats, corr, batch, map_steps, args.map_lr)
                    map_seconds = time.perf_counter() - map_start
                    for method, est in map_estimates.items():
                        record(method.replace('map', 'map_marg'), est, map_seconds)

                    if args.include_full:
                        map_start = time.perf_counter()
                        full_map_estimates = _run_map_full(
                            stats, corr, batch, map_steps, args.map_lr
                        )
                        map_seconds = time.perf_counter() - map_start
                        for method, est in full_map_estimates.items():
                            record(method.replace('map', 'map_full'), est, map_seconds)

            for method in method_order:
                print(stores[method].row(method, data_id, partition, n_total))


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=SIZES, choices=SIZES)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--map-steps', default='5,10,20')
    parser.add_argument('--map-lr', type=float, default=0.03)
    parser.add_argument('--include-full', action='store_true')
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run(setup())
