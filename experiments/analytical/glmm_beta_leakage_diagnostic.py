"""Diagnostic for fixed-effect leakage into Gaussian BLUP residuals.

This is the I5 follow-up to ``metabeta/analytical/plan.md``. It keeps the
estimator unchanged and recomputes observable GLS diagnostics from the returned
``Psi`` and ``sigma_eps_est``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.glmm import glmm
from metabeta.analytical.linalg import (
    _adaptiveRidge,
    _adaptiveRidgeBm,
    _eighWithJitter,
    _safeSolve,
)
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import datasetFilename


SIZES = ['small', 'medium', 'large', 'huge']
BLEND_WEIGHTS = [0.1, 0.25, 0.5, 0.75]


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


def _gls_condition(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    Psi: torch.Tensor,
    sigma_eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    A_gls = (XtX - correction_XX) / sigma_eps.square().clamp(min=1e-12)[:, None, None]
    A_reg = A_gls + _adaptiveRidge(A_gls)
    beta_gls = _safeSolve(
        A_reg,
        (Xty - correction_Xy) / sigma_eps.square().clamp(min=1e-12)[:, None],
    )
    xtx_max_diag = XtX.diagonal(dim1=-1, dim2=-2).amax(dim=-1).clamp(min=1.0)
    beta_mask = (XtX - correction_XX).diagonal(dim1=-2, dim2=-1).abs() > (
        1e-3 * xtx_max_diag[:, None]
    )

    eig = torch.linalg.eigvalsh(0.5 * (A_reg + A_reg.mT)).clamp(min=0.0)
    max_eig = eig.amax(dim=-1).clamp(min=1e-30)
    active_eig = torch.where(eig > max_eig[:, None] * 1e-8, eig, torch.full_like(eig, np.inf))
    min_eig = active_eig.amin(dim=-1).clamp(min=1e-30)
    cond = (max_eig / min_eig).clamp(max=1e30)
    eff_rank = (eig > max_eig[:, None] * 1e-3).sum(dim=-1).clamp(max=d)
    return beta_mask, eff_rank.to(torch.float64), cond


def _projection_error(
    Xm: torch.Tensor,
    mask_n: torch.Tensor,
    beta: torch.Tensor,
    beta_true: torch.Tensor,
) -> torch.Tensor:
    delta = beta - beta_true
    pred_delta = torch.einsum('bmnd,bd->bmn', Xm, delta)
    denom = mask_n.sum(dim=(1, 2)).clamp(min=1.0)
    return ((pred_delta.square() * mask_n).sum(dim=(1, 2)) / denom).sqrt()


def _blup_for_beta(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    beta: torch.Tensor,
    Psi: torch.Tensor,
    sigma_eps: torch.Tensor,
) -> torch.Tensor:
    B, m, _, q = Zm.shape
    eye_q = torch.eye(q, device=Zm.device, dtype=Zm.dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)
    active = mask_m.bool()
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)

    vals, vecs = _eighWithJitter(Psi + sigma_eps.square()[:, None, None] * 1e-4 * eye_q)
    Psi_inv = vecs @ torch.diag_embed(1.0 / vals.clamp(min=1e-30)) @ vecs.mT
    inner = sigma_eps.square()[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm) * mask_m[:, :, None, None]
    resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta)) * mask_n
    ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    return torch.einsum('bmqr,bmr->bmq', W_g, ztr).nan_to_num().clamp(-20.0, 20.0)


def _append_bin_table(
    rows: list[list[str]],
    label: str,
    metric: np.ndarray,
    err: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 4,
    log_metric: bool = False,
) -> None:
    finite = np.isfinite(metric) & np.isfinite(err) & np.isfinite(truth)
    if finite.sum() < 20:
        return
    values = np.log10(metric[finite].clip(min=1e-30)) if log_metric else metric[finite]
    edges = np.unique(np.nanpercentile(values, np.linspace(0, 100, n_bins + 1)))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel_finite = values >= lo
        sel_finite &= values <= hi if i == len(edges) - 2 else values < hi
        if sel_finite.sum() < 10:
            continue
        full_sel = np.zeros_like(finite)
        full_sel[np.flatnonzero(finite)[sel_finite]] = True
        label_range = f'{10**lo:.2g}-{10**hi:.2g}' if log_metric else f'{lo:.3f}-{hi:.3f}'
        rows.append(
            [
                label,
                label_range,
                str(int(full_sel.sum())),
                f'{_nrmse(err[full_sel], truth[full_sel]):.3f}',
            ]
        )


def beta_leakage_suite() -> None:
    print('=== I5 beta leakage diagnostic ===')
    suite_rows = []
    breakdown_rows = []

    for size in SIZES:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']

            beta_est_errs: list[np.ndarray] = []
            beta_ols_errs: list[np.ndarray] = []
            beta_wg_errs: list[np.ndarray] = []
            beta_truths: list[np.ndarray] = []
            blup_errs: list[np.ndarray] = []
            blup_truths: list[np.ndarray] = []
            proj_est: list[np.ndarray] = []
            proj_ols: list[np.ndarray] = []
            proj_wg: list[np.ndarray] = []
            max_beta_err: list[np.ndarray] = []
            beta_mask_count: list[np.ndarray] = []
            beta_mask_any: list[np.ndarray] = []
            beta_rank: list[np.ndarray] = []
            cond: list[np.ndarray] = []

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

                        beta_true = batch['ffx']
                        beta_est = stats['beta_est']
                        beta_wg = stats['beta_wg']
                        beta_ols = _beta_ols(batch['X'], batch['y'], mask_n)
                        bm, erank, cnd = _gls_condition(
                            batch['X'],
                            batch['y'],
                            Zm,
                            mask_n,
                            mask_m,
                            stats['Psi'],
                            stats['sigma_eps_est'].squeeze(-1),
                        )

                        active_d = batch['mask_d'].bool()
                        active_m = batch['mask_m'].bool()
                        active_q = batch['mask_q'][..., :max_q].bool()
                        for b in range(batch['X'].shape[0]):
                            d_sel = active_d[b]
                            q_sel = active_q[b]
                            m_sel = active_m[b]
                            truth = batch['rfx'][b][m_sel][:, q_sel].reshape(-1).numpy()
                            err = (
                                stats['blup_est'][b][m_sel][:, q_sel].reshape(-1)
                                - torch.tensor(truth)
                            ).numpy()
                            reps = int(truth.size)

                            be_est = (beta_est[b, d_sel] - beta_true[b, d_sel]).numpy()
                            be_ols = (beta_ols[b, d_sel] - beta_true[b, d_sel]).numpy()
                            be_wg = (beta_wg[b, d_sel] - beta_true[b, d_sel]).numpy()
                            beta_est_errs.append(be_est)
                            beta_ols_errs.append(be_ols)
                            beta_wg_errs.append(be_wg)
                            beta_truths.append(beta_true[b, d_sel].numpy())
                            blup_errs.append(err)
                            blup_truths.append(truth)

                            p_est = float(
                                _projection_error(
                                    batch['X'][b : b + 1],
                                    mask_n[b : b + 1],
                                    beta_est[b : b + 1],
                                    beta_true[b : b + 1],
                                )[0]
                            )
                            p_ols = float(
                                _projection_error(
                                    batch['X'][b : b + 1],
                                    mask_n[b : b + 1],
                                    beta_ols[b : b + 1],
                                    beta_true[b : b + 1],
                                )[0]
                            )
                            p_wg = float(
                                _projection_error(
                                    batch['X'][b : b + 1],
                                    mask_n[b : b + 1],
                                    beta_wg[b : b + 1],
                                    beta_true[b : b + 1],
                                )[0]
                            )
                            proj_est.append(np.full(reps, p_est))
                            proj_ols.append(np.full(reps, p_ols))
                            proj_wg.append(np.full(reps, p_wg))
                            max_beta_err.append(
                                np.full(
                                    reps,
                                    float((beta_est[b, d_sel] - beta_true[b, d_sel]).abs().max()),
                                )
                            )
                            beta_mask_count.append(np.full(reps, int((bm[b] & d_sel).sum())))
                            beta_mask_any.append(np.full(reps, int(bool((bm[b] & d_sel).any()))))
                            beta_rank.append(np.full(reps, float(erank[b])))
                            cond.append(np.full(reps, float(cnd[b])))

            beta_true_flat = np.concatenate(beta_truths)
            beta_est_flat = np.concatenate(beta_est_errs)
            beta_ols_flat = np.concatenate(beta_ols_errs)
            beta_wg_flat = np.concatenate(beta_wg_errs)
            blup_err_flat = np.concatenate(blup_errs)
            blup_truth_flat = np.concatenate(blup_truths)
            proj_est_flat = np.concatenate(proj_est)
            proj_ols_flat = np.concatenate(proj_ols)
            proj_wg_flat = np.concatenate(proj_wg)
            max_beta_flat = np.concatenate(max_beta_err)
            mask_count_flat = np.concatenate(beta_mask_count)
            mask_any_flat = np.concatenate(beta_mask_any)
            rank_flat = np.concatenate(beta_rank)
            cond_flat = np.concatenate(cond)

            suite_rows.append(
                [
                    data_id,
                    partition,
                    f'{_nrmse(beta_est_flat, beta_true_flat):.3f}',
                    f'{np.mean(beta_est_flat):+.3f}',
                    f'{_nrmse(beta_ols_flat, beta_true_flat):.3f}',
                    f'{_nrmse(beta_wg_flat, beta_true_flat):.3f}',
                    f'{_nrmse(blup_err_flat, blup_truth_flat):.3f}',
                    f'{np.median(proj_est_flat):.3f}',
                    f'{np.median(proj_ols_flat):.3f}',
                    f'{np.median(proj_wg_flat):.3f}',
                    f'{np.median(cond_flat):.2g}',
                    f'{np.median(rank_flat):.1f}',
                    f'{np.mean(mask_count_flat):.1f}',
                ]
            )
            _append_bin_table(
                breakdown_rows,
                f'{data_id}/{partition} max_beta',
                max_beta_flat,
                blup_err_flat,
                blup_truth_flat,
            )
            _append_bin_table(
                breakdown_rows,
                f'{data_id}/{partition} proj_est',
                proj_est_flat,
                blup_err_flat,
                blup_truth_flat,
            )
            _append_bin_table(
                breakdown_rows,
                f'{data_id}/{partition} cond',
                cond_flat,
                blup_err_flat,
                blup_truth_flat,
                log_metric=True,
            )
            _append_bin_table(
                breakdown_rows,
                f'{data_id}/{partition} beta_rank',
                rank_flat,
                blup_err_flat,
                blup_truth_flat,
            )
            _append_bin_table(
                breakdown_rows,
                f'{data_id}/{partition} beta_mask_count',
                mask_count_flat,
                blup_err_flat,
                blup_truth_flat,
            )
            _append_bin_table(
                breakdown_rows,
                f'{data_id}/{partition} beta_mask_any',
                mask_any_flat,
                blup_err_flat,
                blup_truth_flat,
            )

    print(
        tabulate(
            suite_rows,
            headers=[
                'dataset',
                'part',
                'beta_est',
                'bias',
                'beta_ols',
                'beta_wg',
                'BLUP',
                'med proj est',
                'med proj ols',
                'med proj wg',
                'med cond',
                'med rank',
                'mean mask',
            ],
            tablefmt='simple',
        )
    )
    print('\n=== BLUP NRMSE by beta diagnostic bins ===')
    print(tabulate(breakdown_rows, headers=['slice', 'bin', 'N', 'BLUP'], tablefmt='simple'))


def beta_ablation_small_mixed() -> None:
    print('\n=== I5 small-n-mixed beta ablations ===')
    data_id = 'small-n-mixed'
    cfg = loadDataConfig(data_id)
    max_q = cfg['max_q']
    cases = {'beta_est baseline': []}
    cases['beta_ols'] = []
    for alpha in BLEND_WEIGHTS:
        cases[f'blend ols {alpha:g}'] = []
    cases['oracle best est/ols'] = []
    truth_parts: list[np.ndarray] = []

    with torch.no_grad():
        for path in _paths(data_id, 'train', 2):
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
                beta_est = stats['beta_est']
                beta_ols = _beta_ols(batch['X'], batch['y'], mask_n)
                sigma_eps = stats['sigma_eps_est'].squeeze(-1)
                blups = {
                    'beta_est baseline': stats['blup_est'],
                    'beta_ols': _blup_for_beta(
                        batch['X'],
                        batch['y'],
                        Zm,
                        mask_n,
                        mask_m,
                        beta_ols,
                        stats['Psi'],
                        sigma_eps,
                    ),
                }
                for alpha in BLEND_WEIGHTS:
                    beta_blend = (1.0 - alpha) * beta_est + alpha * beta_ols
                    blups[f'blend ols {alpha:g}'] = _blup_for_beta(
                        batch['X'],
                        batch['y'],
                        Zm,
                        mask_n,
                        mask_m,
                        beta_blend,
                        stats['Psi'],
                        sigma_eps,
                    )

                active_m = batch['mask_m'].bool()
                active_q = batch['mask_q'][..., :max_q].bool()
                for b in range(batch['X'].shape[0]):
                    truth = batch['rfx'][b][active_m[b]][:, active_q[b]].reshape(-1).numpy()
                    truth_parts.append(truth)
                    best_err: np.ndarray | None = None
                    for name, value in blups.items():
                        est = value[b][active_m[b]][:, active_q[b]].reshape(-1).numpy()
                        err = est - truth
                        cases[name].append(err)
                        if name in ('beta_est baseline', 'beta_ols'):
                            if best_err is None or np.mean(err**2) < np.mean(best_err**2):
                                best_err = err
                    cases['oracle best est/ols'].append(
                        best_err if best_err is not None else np.zeros_like(truth)
                    )

    truth_all = np.concatenate(truth_parts)
    rows = []
    for name, parts in cases.items():
        err = np.concatenate(parts)
        rows.append([name, f'{_nrmse(err, truth_all):.4f}', f'{float(np.mean(err)):+.4f}'])
    print(tabulate(rows, headers=['case', 'BLUP NRMSE', 'bias'], tablefmt='simple'))


def beta_ablation_suite() -> None:
    print('\n=== I5 beta ablation suite ===')
    cases = ['beta_est', 'beta_ols']
    cases.extend(f'blend {alpha:g}' for alpha in BLEND_WEIGHTS)
    rows = []

    for size in SIZES:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            err_parts = {case: [] for case in cases}
            truth_parts: list[np.ndarray] = []

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
                        beta_est = stats['beta_est']
                        beta_ols = _beta_ols(batch['X'], batch['y'], mask_n)
                        sigma_eps = stats['sigma_eps_est'].squeeze(-1)
                        blups = {
                            'beta_est': stats['blup_est'],
                            'beta_ols': _blup_for_beta(
                                batch['X'],
                                batch['y'],
                                Zm,
                                mask_n,
                                mask_m,
                                beta_ols,
                                stats['Psi'],
                                sigma_eps,
                            ),
                        }
                        for alpha in BLEND_WEIGHTS:
                            beta_blend = (1.0 - alpha) * beta_est + alpha * beta_ols
                            blups[f'blend {alpha:g}'] = _blup_for_beta(
                                batch['X'],
                                batch['y'],
                                Zm,
                                mask_n,
                                mask_m,
                                beta_blend,
                                stats['Psi'],
                                sigma_eps,
                            )

                        active_m = batch['mask_m'].bool()
                        active_q = batch['mask_q'][..., :max_q].bool()
                        for b in range(batch['X'].shape[0]):
                            truth = batch['rfx'][b][active_m[b]][:, active_q[b]].reshape(-1).numpy()
                            truth_parts.append(truth)
                            for name, value in blups.items():
                                est = value[b][active_m[b]][:, active_q[b]].reshape(-1).numpy()
                                err_parts[name].append(est - truth)

            truth_all = np.concatenate(truth_parts)
            row = [data_id, partition]
            for case in cases:
                row.append(f'{_nrmse(np.concatenate(err_parts[case]), truth_all):.4f}')
            rows.append(row)

    print(tabulate(rows, headers=['dataset', 'part'] + cases, tablefmt='simple'))


if __name__ == '__main__':
    beta_leakage_suite()
    beta_ablation_small_mixed()
    beta_ablation_suite()
