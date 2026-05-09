"""Diagnostic for observable adaptive-alpha gates in Gaussian BLUP residuals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.glmm import glmm
from metabeta.analytical.linalg import _adaptiveRidge, _adaptiveRidgeBm, _eighWithJitter, _safeSolve
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


def _beta_ols(Xm: torch.Tensor, ym: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
    X_masked = Xm * mask_n[..., None]
    XtX = torch.einsum('bmnd,bmnk->bdk', X_masked, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', X_masked, ym)
    return _safeSolve(XtX + _adaptiveRidge(XtX), Xty).nan_to_num()


def _gls_diagnostics(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_m: torch.Tensor,
    Psi: torch.Tensor,
    sigma_eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, m, _, q = Zm.shape
    d = Xm.shape[-1]
    active = mask_m.bool()
    eye_q = torch.eye(q, device=Zm.device, dtype=Zm.dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)

    vals, vecs = _eighWithJitter(Psi + sigma_eps.square()[:, None, None] * 1e-4 * eye_q)
    Psi_inv = vecs @ torch.diag_embed(1.0 / vals.clamp(min=1e-30)) @ vecs.mT
    inner = sigma_eps.square()[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm) * mask_m[:, :, None, None]

    W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)
    correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)
    W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)
    correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)
    se2 = sigma_eps.square().clamp(min=1e-12)
    A_gls = (XtX - correction_XX) / se2[:, None, None]
    A_reg = A_gls + _adaptiveRidge(A_gls)

    xtx_max_diag = XtX.diagonal(dim1=-1, dim2=-2).amax(dim=-1).clamp(min=1.0)
    beta_mask = (XtX - correction_XX).diagonal(dim1=-2, dim2=-1).abs() > (
        1e-3 * xtx_max_diag[:, None]
    )
    eig = torch.linalg.eigvalsh(0.5 * (A_reg + A_reg.mT)).clamp(min=0.0)
    max_eig = eig.amax(dim=-1).clamp(min=1e-30)
    eff_rank = (eig > max_eig[:, None] * 1e-3).sum(dim=-1).clamp(max=d)
    active_eig = torch.where(eig > max_eig[:, None] * 1e-8, eig, torch.full_like(eig, np.inf))
    cond = max_eig / active_eig.amin(dim=-1).clamp(min=1e-30)
    return W_g, beta_mask, eff_rank.to(torch.float64), cond.clamp(max=1e30)


def _candidate_alphas(
    beta_mask_count: torch.Tensor,
    eff_rank: torch.Tensor,
    cond: torch.Tensor,
    d_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dtype = cond.dtype
    device = eff_rank.device
    d_safe = d_count.to(dtype).clamp(min=1.0)
    mask_frac = beta_mask_count.to(dtype) / d_safe
    rank_frac = eff_rank / d_safe
    weak_any = (mask_frac < 1.0) | (rank_frac < 1.0)
    weak_deep = (mask_frac <= 0.75) | (rank_frac <= 0.75)
    weak_rank = rank_frac < 0.875
    weak_mask = mask_frac < 0.875
    poor_mid_cond = (cond > 1e3) & (cond < 1.2e6)
    low_d = d_count <= 4
    mid_d = (d_count > 4) & (d_count <= 8)

    def full(value: float) -> torch.Tensor:
        return torch.full_like(eff_rank, value, dtype=dtype, device=device)

    def choose(weak: torch.Tensor, weak_alpha: float, strong_alpha: float) -> torch.Tensor:
        return torch.where(weak, full(weak_alpha), full(strong_alpha))

    return {
        'scalar_0.25': full(0.25),
        'scalar_0.50': full(0.50),
        'scalar_0.65': full(0.65),
        'scalar_0.75': full(0.75),
        'scalar_1.00': full(1.00),
        'weak_any_075_else_050': choose(weak_any, 0.75, 0.50),
        'weak_any_100_else_050': choose(weak_any, 1.00, 0.50),
        'weak_deep_100_else_050': choose(weak_deep, 1.00, 0.50),
        'rank_lt_0875_100_else_050': choose(weak_rank, 1.00, 0.50),
        'mask_lt_0875_100_else_050': choose(weak_mask, 1.00, 0.50),
        'weak_any_075_condmid_050': torch.where(
            poor_mid_cond,
            full(0.50),
            choose(weak_any, 0.75, 0.50),
        ),
        'weak_any_100_condmid_050': torch.where(
            poor_mid_cond,
            full(0.50),
            choose(weak_any, 1.00, 0.50),
        ),
        'd_le4_100_else_075': choose(low_d, 1.00, 0.75),
        'd_le4_100_else_065': choose(low_d, 1.00, 0.65),
        'd_le4_100_d_le8_065_else_075': torch.where(
            low_d,
            full(1.00),
            torch.where(mid_d, full(0.65), full(0.75)),
        ),
        'd_le4_100_d_le8_050_else_075': torch.where(
            low_d,
            full(1.00),
            torch.where(mid_d, full(0.50), full(0.75)),
        ),
    }


def run_alpha_gate_diagnostic() -> None:
    err_parts: dict[str, dict[str, list[np.ndarray]]] = {}
    truth_parts: dict[str, list[np.ndarray]] = {}

    for size in SIZES:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            key = f'{data_id}/{partition}'
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            truth_parts[key] = []

            with torch.no_grad():
                for path in _paths(data_id, partition, n_epochs):
                    for batch in Dataloader(path, batch_size=32, shuffle=False):
                        batch = toDevice(batch, torch.device('cpu'))
                        Zm = batch['Z'][..., :max_q]
                        mask_n = batch['mask_n'].float()
                        mask_m = batch['mask_m'].float()
                        stats = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            mask_n,
                            mask_m,
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            eta_rfx=batch.get('eta_rfx'),
                            mask_q=batch.get('mask_q'),
                        )
                        beta_ols = _beta_ols(batch['X'], batch['y'], mask_n)
                        W_g, beta_mask, eff_rank, cond = _gls_diagnostics(
                            batch['X'],
                            batch['y'],
                            Zm,
                            mask_m,
                            stats['Psi'],
                            stats['sigma_eps_est'].squeeze(-1),
                        )
                        d_count = batch['mask_d'].bool().sum(dim=-1)
                        alphas = _candidate_alphas(
                            beta_mask.sum(dim=-1),
                            eff_rank,
                            cond,
                            d_count,
                        )

                        active_m = batch['mask_m'].bool()
                        active_q = batch['mask_q'][..., :max_q].bool()
                        for name, alpha in alphas.items():
                            err_parts.setdefault(name, {}).setdefault(key, [])
                            beta = (1.0 - alpha[:, None]) * stats['beta_est'] + alpha[
                                :, None
                            ] * beta_ols
                            resid = (
                                batch['y'] - torch.einsum('bmnd,bd->bmn', batch['X'], beta)
                            ) * mask_n
                            ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
                            blups = torch.einsum('bmqr,bmr->bmq', W_g, ztr)
                            blups = blups.nan_to_num().clamp(-20.0, 20.0)
                            for b in range(batch['X'].shape[0]):
                                truth = batch['rfx'][b][active_m[b]][:, active_q[b]].reshape(-1)
                                est = blups[b][active_m[b]][:, active_q[b]].reshape(-1)
                                err_parts[name][key].append((est - truth).cpu().numpy())

                        for b in range(batch['X'].shape[0]):
                            truth = batch['rfx'][b][active_m[b]][:, active_q[b]].reshape(-1)
                            truth_parts[key].append(truth.cpu().numpy())

    keys = list(truth_parts)
    rows = []
    for name, by_key in err_parts.items():
        values = []
        worst_vs_i6 = 0.0
        mean_blup = 0.0
        for key in keys:
            truth = np.concatenate(truth_parts[key])
            err = np.concatenate(by_key[key])
            score = _nrmse(err, truth)
            values.append(score)
            mean_blup += score / len(keys)
            i6 = _nrmse(np.concatenate(err_parts['scalar_0.75'][key]), truth)
            worst_vs_i6 = max(worst_vs_i6, score / i6 - 1.0)
        rows.append(
            [
                name,
                f'{mean_blup:.4f}',
                f'{worst_vs_i6:+.2%}',
                *[f'{value:.4f}' for value in values],
            ]
        )

    print(
        tabulate(
            rows,
            headers=['candidate', 'mean', 'worst_vs_i6'] + keys,
            tablefmt='simple',
        )
    )


if __name__ == '__main__':
    run_alpha_gate_diagnostic()
