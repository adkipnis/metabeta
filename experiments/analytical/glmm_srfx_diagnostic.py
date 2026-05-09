"""Diagnostic for Gaussian GLMM random-effect scale estimates.

Runs the required analytical suite and traces the normal estimator internals without
patching them. The goal is to separate sigma(RFX) error into initialization, EM
updates, floor/cap activity, component fallback, and off-diagonal covariance effects.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.constants import _NORMAL_FULL_MIN_EM
from metabeta.analytical.linalg import (
    _adaptiveRidge,
    _adaptiveRidgeBm,
    _groupZDiagnostics,
    _psdProject,
    _safeSolve,
    _shrinkOffDiagonal,
)
from metabeta.analytical.normal import (
    _componentwisePsiDiagSignal,
    _emRefineNormal,
    _estimateWithinGroupVariance,
    _forceDiagonalPsi,
    _initialPsiMom,
    _normalGlsAndBlups,
)
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import datasetFilename


SIZES = ['small', 'medium', 'large', 'huge']


@dataclass
class _Trace:
    sigma_init: torch.Tensor
    sigma_post_em: torch.Tensor
    sigma_final: torch.Tensor
    sigma_em_raw: torch.Tensor
    sigma_em_winsor: torch.Tensor
    sigma_em_trim: torch.Tensor
    psi_diag_floor: torch.Tensor
    psi_eig_cap: torch.Tensor
    component_count: torch.Tensor
    mom_mask: torch.Tensor
    enough_full_mom: torch.Tensor
    enough_diag_mom: torch.Tensor
    use_component_diag: torch.Tensor
    floor_hit_init: torch.Tensor
    floor_hit_post_em: torch.Tensor
    cap_hit_init: torch.Tensor
    cap_hit_post_em: torch.Tensor
    corr_alpha: torch.Tensor
    corr_post_em: torch.Tensor
    corr_final: torch.Tensor


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    data_dir = ROOT / 'metabeta' / 'outputs' / 'data' / cfg['data_id']
    if partition == 'train':
        return [data_dir / datasetFilename('train', ep) for ep in range(1, n_epochs + 1)]
    return [data_dir / f'{partition}.npz']


def _safe_corr(Psi: torch.Tensor) -> torch.Tensor:
    std = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12).sqrt()
    return (Psi / (std[:, :, None] * std[:, None, :]).clamp(min=1e-12)).clamp(-1.0, 1.0)


def _true_psi(batch: dict[str, torch.Tensor], max_q: int) -> torch.Tensor:
    sigma = batch['sigma_rfx'][..., :max_q]
    corr = batch['corr_rfx'][..., :max_q, :max_q]
    eta = batch.get('eta_rfx')
    if eta is not None:
        eye = torch.eye(max_q, device=sigma.device, dtype=sigma.dtype)
        corr = torch.where(eta[:, None, None] == 0, eye.expand_as(corr), corr)
    return corr * sigma[:, :, None] * sigma[:, None, :]


def _em_targets(
    Psi: torch.Tensor,
    se2: torch.Tensor,
    gls,
    mom_mask: torch.Tensor,
    G_mom: torch.Tensor,
    enough_full_mom: torch.Tensor,
    active_qq: torch.Tensor,
    psi_diag_floor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return first EM M-step targets before winsor, after winsor, and after trim."""
    mom4 = mom_mask[:, :, None, None]
    post_cov = se2[:, None, None, None] * gls.W_g

    blup_outer_raw = torch.einsum('bmq,bmr->bmqr', gls.blups, gls.blups)
    Psi_raw = _psdProject(((blup_outer_raw + post_cov) * mom4).sum(dim=1) / G_mom[:, None, None])

    psi_diag_cur = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor).sqrt()
    blup_cap = 10.0 * psi_diag_cur[:, None, :]
    blups_winsor = gls.blups.clamp(min=-blup_cap, max=blup_cap)
    blup_outer_winsor = torch.einsum('bmq,bmr->bmqr', blups_winsor, blups_winsor)
    Psi_winsor = _psdProject(
        ((blup_outer_winsor + post_cov) * mom4).sum(dim=1) / G_mom[:, None, None]
    )

    blup_norm = blups_winsor.square().sum(dim=-1)
    mom_norm_mean = (blup_norm * mom_mask).sum(dim=1) / G_mom
    trim_mask = mom_mask.bool() & (blup_norm <= 3.0 * mom_norm_mean[:, None])
    trim_count = trim_mask.float().sum(dim=1).clamp(min=1.0)
    Psi_trim = _psdProject(
        ((blup_outer_winsor + post_cov) * trim_mask[:, :, None, None]).sum(dim=1)
        / trim_count[:, None, None]
    )

    def finish(Psi_target: torch.Tensor) -> torch.Tensor:
        psi_diag = Psi_target.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)
        Psi_diag_safe = torch.where(
            enough_full_mom[:, None, None],
            Psi_target,
            torch.diag_embed(psi_diag),
        )
        return (Psi_diag_safe * active_qq).diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()

    return finish(Psi_raw), finish(Psi_winsor), finish(Psi_trim)


def _trace_normal(batch: dict[str, torch.Tensor], max_q: int) -> _Trace:
    Xm = batch['X']
    ym = batch['y']
    Zm = batch['Z'][..., :max_q]
    mask_n = batch['mask_n'].float()
    mask_m = batch['mask_m'].float()
    ns = batch['ns'].clamp(min=1).float()
    n_total = batch['n'].float()
    mask_q = batch.get('mask_q')
    if mask_q is not None:
        Zm = Zm * mask_q[:, None, None, :max_q].to(device=Zm.device, dtype=Zm.dtype)
    uncorr = (batch['eta_rfx'] == 0) if 'eta_rfx' in batch else None

    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    G = mask_m.sum(dim=1).clamp(min=1.0)
    active = mask_m.bool()
    mask4 = mask_m[:, :, None, None]
    eye_q = torch.eye(q, device=Xm.device, dtype=Xm.dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)

    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)
    mom_mask, z_rank, _, active_components, active_count = _groupZDiagnostics(ZtZ, mask_m, ns, q)
    G_mom_raw = mom_mask.sum(dim=1)
    G_mom = G_mom_raw.clamp(min=1.0)
    enough_full_mom = G_mom_raw >= torch.maximum(
        active_count + 1.0, G_mom_raw.new_full((B,), float(d + 1))
    )
    enough_diag_mom = (G_mom_raw >= 2.0) & (active_count > 0)
    active_q = active_components.to(Zm.dtype)
    active_qq = active_q[:, :, None] * active_q[:, None, :]

    ZtZ_inv = _safeSolve(ZtZ_safe + _adaptiveRidgeBm(ZtZ_safe), eye_q_bm)
    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)
    beta_wg, sigma_eps_sq, mx_rank = _estimateWithinGroupVariance(
        Xm, ym, mask_n, n_total, Zm, ZtZ_inv, ZtX, z_rank
    )
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    beta_ols, _, Psi_init, psi_diag_floor, psi_eig_cap = _initialPsiMom(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        ZtZ_safe,
        ZtZ_inv,
        XtX,
        Xty,
        sigma_eps_sq,
        mom_mask,
        G_mom,
        enough_full_mom,
        enough_diag_mom,
        active_q,
        active_qq,
        eye_q,
    )
    Psi_init = _forceDiagonalPsi(Psi_init, uncorr).nan_to_num(nan=0.0, posinf=0.0)
    se2 = sigma_eps_sq.clamp(min=1e-12)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)
    beta_gls_fallback = torch.where(mx_rank[:, None] > 0, beta_wg, beta_ols)
    gls = _normalGlsAndBlups(
        Xm,
        ym,
        Zm,
        mask_n,
        ZtZ_safe,
        Zty,
        ZtX,
        XtX,
        Xty,
        XtZ,
        Psi_init,
        se2,
        eye_q,
        eye_q_bm,
        mask4,
        beta_gls_fallback,
    )
    weak_multi_q = (active_count > 1.0) & ~enough_full_mom
    beta_mask = torch.where(
        weak_multi_q[:, None],
        gls.beta_mask,
        gls.beta_mask.any(dim=-1, keepdim=True),
    )
    beta_rank = mx_rank.clamp(min=1.0, max=float(d))
    sigma_em_raw, sigma_em_winsor, sigma_em_trim = _em_targets(
        Psi_init,
        se2,
        gls,
        mom_mask,
        G_mom,
        enough_full_mom,
        active_qq,
        psi_diag_floor,
    )

    Psi_post_em, _, _ = _emRefineNormal(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        ZtZ_safe,
        Zty,
        ZtX,
        XtX,
        Xty,
        XtZ,
        eye_q,
        eye_q_bm,
        mask4,
        mom_mask[:, :, None, None],
        G_mom,
        enough_full_mom,
        active_qq,
        beta_wg,
        beta_rank,
        beta_mask,
        Psi_init,
        se2,
        psi_diag_floor,
        psi_eig_cap,
        gls,
        max(3, _NORMAL_FULL_MIN_EM),
        uncorr,
    )
    corr_alpha = G / (G + 5.0)
    Psi_final = _shrinkOffDiagonal(Psi_post_em, corr_alpha)

    resid_full = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_ols)) * mask_n
    _, component_count = _componentwisePsiDiagSignal(
        Zm, resid_full, mask_m, ns, sigma_eps_sq, active_q
    )
    use_component_diag = (component_count >= 5.0) & ~(G_mom_raw > 0)[:, None]

    init_diag = Psi_init.diagonal(dim1=-2, dim2=-1)
    post_diag = Psi_post_em.diagonal(dim1=-2, dim2=-1)
    floor_tol = torch.maximum(psi_diag_floor.abs() * 1e-3, psi_diag_floor.new_full((), 1e-10))
    floor_hit_init = init_diag <= psi_diag_floor + floor_tol
    floor_hit_post_em = post_diag <= psi_diag_floor + floor_tol
    eig_init = torch.linalg.eigvalsh(0.5 * (Psi_init + Psi_init.mT)).amax(dim=-1)
    eig_post = torch.linalg.eigvalsh(0.5 * (Psi_post_em + Psi_post_em.mT)).amax(dim=-1)
    cap_hit_init = eig_init >= 0.999 * psi_eig_cap
    cap_hit_post_em = eig_post >= 0.999 * psi_eig_cap

    return _Trace(
        sigma_init=init_diag.clamp(min=0.0).sqrt(),
        sigma_post_em=post_diag.clamp(min=0.0).sqrt(),
        sigma_final=Psi_final.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt(),
        sigma_em_raw=sigma_em_raw,
        sigma_em_winsor=sigma_em_winsor,
        sigma_em_trim=sigma_em_trim,
        psi_diag_floor=psi_diag_floor,
        psi_eig_cap=psi_eig_cap,
        component_count=component_count,
        mom_mask=mom_mask,
        enough_full_mom=enough_full_mom,
        enough_diag_mom=enough_diag_mom,
        use_component_diag=use_component_diag,
        floor_hit_init=floor_hit_init,
        floor_hit_post_em=floor_hit_post_em,
        cap_hit_init=cap_hit_init,
        cap_hit_post_em=cap_hit_post_em,
        corr_alpha=corr_alpha,
        corr_post_em=_safe_corr(Psi_post_em),
        corr_final=_safe_corr(Psi_final),
    )


def _append_by_active_component(
    values: dict[str, list[float]],
    batch: dict[str, torch.Tensor],
    trace: _Trace,
    key: str,
    row_key: str,
    tensor: torch.Tensor,
) -> None:
    mask_q = batch['mask_q'][..., : tensor.shape[-1]].bool()
    for b in range(mask_q.shape[0]):
        active_q = mask_q[b]
        if not bool(active_q.any()):
            continue
        values[f'{row_key}:{key}'].extend(tensor[b][active_q].detach().cpu().tolist())
        if key != 'init':
            continue
        values[f'{row_key}:truth'].extend(
            batch['sigma_rfx'][b, : tensor.shape[-1]][active_q].cpu().tolist()
        )
        values[f'{row_key}:d'].extend([float(batch['mask_d'][b].sum())] * int(active_q.sum()))
        values[f'{row_key}:q'].extend([float(active_q.sum())] * int(active_q.sum()))
        values[f'{row_key}:G'].extend([float(batch['mask_m'][b].sum())] * int(active_q.sum()))
        values[f'{row_key}:G_mom'].extend([float(trace.mom_mask[b].sum())] * int(active_q.sum()))
        values[f'{row_key}:component_count'].extend(
            trace.component_count[b][active_q].detach().cpu().tolist()
        )


def _bin_label(value: float, edges: np.ndarray) -> str:
    if len(edges) == 1:
        return f'{edges[0]:.2f}'
    for lo, hi in zip(edges[:-1], edges[1:]):
        if value <= hi:
            return f'{lo:.2f}-{hi:.2f}'
    return f'{edges[-2]:.2f}-{edges[-1]:.2f}'


def _summarize_bin(
    name: str,
    records: list[dict[str, float | str | bool]],
    mask_fn,
) -> list[str] | None:
    selected = [r for r in records if mask_fn(r)]
    if not selected:
        return None
    truth = np.array([float(r['truth']) for r in selected])
    err = np.array([float(r['final']) - float(r['truth']) for r in selected])
    rel = np.array(
        [
            (float(r['final']) - float(r['truth'])) / max(abs(float(r['truth'])), 1e-8)
            for r in selected
        ]
    )
    return [
        name,
        str(len(selected)),
        f'{_nrmse(err, truth):.4f}',
        f'{rel.mean():+.3f}',
        f'{_rmse(err):.4f}',
    ]


def run_srfx_diagnostic(
    only_data_id: str | None = None,
    only_partition: str | None = None,
    batch_size: int = 32,
) -> None:
    rows = []
    corr_rows = []
    records: list[dict[str, float | str | bool]] = []

    combos: list[tuple[str, str, int]] = []
    if only_data_id is not None:
        if only_partition is None:
            raise ValueError('--partition is required when --data-id is set')
        combos = [(only_data_id, only_partition, 2 if only_partition == 'train' else 0)]
    else:
        for size in SIZES:
            combos.append((f'{size}-n-mixed', 'train', 2))
            combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])

    with torch.no_grad():
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            by_name: dict[str, list[float]] = defaultdict(list)
            corr_post_err: list[float] = []
            corr_final_err: list[float] = []
            corr_truth: list[float] = []
            n_rows = 0

            for path in _paths(data_id, partition, n_epochs):
                for batch in Dataloader(path, batch_size=batch_size, shuffle=False):
                    batch = toDevice(batch, torch.device('cpu'))
                    trace = _trace_normal(batch, max_q)
                    n_rows += batch['X'].shape[0]

                    for name, tensor in [
                        ('init', trace.sigma_init),
                        ('post_em', trace.sigma_post_em),
                        ('final', trace.sigma_final),
                        ('em_raw', trace.sigma_em_raw),
                        ('em_winsor', trace.sigma_em_winsor),
                        ('em_trim', trace.sigma_em_trim),
                    ]:
                        _append_by_active_component(by_name, batch, trace, name, 'srfx', tensor)

                    mask_q = batch['mask_q'][..., :max_q].bool()
                    true_Psi = _true_psi(batch, max_q)
                    true_corr = _safe_corr(true_Psi)
                    for b in range(batch['X'].shape[0]):
                        active_q = mask_q[b]
                        q_count = int(active_q.sum())
                        if q_count < 1:
                            continue
                        active_idx = active_q.nonzero(as_tuple=False).flatten()
                        d_count = int(batch['mask_d'][b].sum())
                        G = float(batch['mask_m'][b].sum())
                        G_mom = float(trace.mom_mask[b].sum())
                        path_label = 'fallback'
                        if bool(trace.enough_full_mom[b]):
                            path_label = 'full_mom'
                        elif bool(trace.enough_diag_mom[b]):
                            path_label = 'diag_mom'

                        for q_idx in active_idx.tolist():
                            component_path = path_label
                            if bool(trace.use_component_diag[b, q_idx]):
                                component_path = 'component_diag'
                            truth = float(batch['sigma_rfx'][b, q_idx])
                            final = float(trace.sigma_final[b, q_idx])
                            records.append(
                                {
                                    'dataset': f'{data_id}/{partition}',
                                    'truth': truth,
                                    'final': final,
                                    'init': float(trace.sigma_init[b, q_idx]),
                                    'post_em': float(trace.sigma_post_em[b, q_idx]),
                                    'd': float(d_count),
                                    'q': float(q_count),
                                    'G': G,
                                    'G_mom': G_mom,
                                    'component_count': float(trace.component_count[b, q_idx]),
                                    'path': component_path,
                                    'floor_init': bool(trace.floor_hit_init[b, q_idx]),
                                    'floor_post_em': bool(trace.floor_hit_post_em[b, q_idx]),
                                    'cap_init': bool(trace.cap_hit_init[b]),
                                    'cap_post_em': bool(trace.cap_hit_post_em[b]),
                                }
                            )

                        if q_count >= 2:
                            ti = torch.tril_indices(q_count, q_count, offset=-1)
                            corr_post = (
                                trace.corr_post_em[b]
                                .index_select(0, active_idx)
                                .index_select(1, active_idx)
                            )
                            corr_final = (
                                trace.corr_final[b]
                                .index_select(0, active_idx)
                                .index_select(1, active_idx)
                            )
                            corr_true = (
                                true_corr[b].index_select(0, active_idx).index_select(1, active_idx)
                            )
                            corr_post_err.extend(
                                (corr_post[ti[0], ti[1]] - corr_true[ti[0], ti[1]]).cpu().tolist()
                            )
                            corr_final_err.extend(
                                (corr_final[ti[0], ti[1]] - corr_true[ti[0], ti[1]]).cpu().tolist()
                            )
                            corr_truth.extend(corr_true[ti[0], ti[1]].cpu().tolist())

            truth = np.array(by_name['srfx:truth'])
            final = np.array(by_name['srfx:final'])
            init = np.array(by_name['srfx:init'])
            post_em = np.array(by_name['srfx:post_em'])
            em_raw = np.array(by_name['srfx:em_raw'])
            em_winsor = np.array(by_name['srfx:em_winsor'])
            em_trim = np.array(by_name['srfx:em_trim'])
            rel = (final - truth) / np.maximum(np.abs(truth), 1e-8)

            dataset_records = [r for r in records if r['dataset'] == f'{data_id}/{partition}']
            rows.append(
                [
                    data_id,
                    partition,
                    str(n_rows),
                    f'{_nrmse(final - truth, truth):.4f}',
                    f'{rel.mean():+.3f}',
                    f'{_nrmse(init - truth, truth):.4f}',
                    f'{_nrmse(post_em - truth, truth):.4f}',
                    f'{_nrmse(em_raw - truth, truth):.4f}',
                    f'{_nrmse(em_winsor - truth, truth):.4f}',
                    f'{_nrmse(em_trim - truth, truth):.4f}',
                    f'{np.mean([r["floor_post_em"] for r in dataset_records]):.1%}',
                    f'{np.mean([r["cap_post_em"] for r in dataset_records]):.1%}',
                    f'{np.mean(by_name["srfx:G_mom"]):.1f}',
                ]
            )

            if corr_truth:
                corr_truth_np = np.array(corr_truth)
                corr_rows.append(
                    [
                        data_id,
                        partition,
                        str(len(corr_truth)),
                        f'{_nrmse(np.array(corr_post_err), corr_truth_np):.4f}',
                        f'{_nrmse(np.array(corr_final_err), corr_truth_np):.4f}',
                        f'{_rmse(np.array(corr_final_err)):.4f}',
                    ]
                )

    print('\nPer dataset sigma(RFX) trace')
    print(
        tabulate(
            rows,
            headers=[
                'dataset',
                'part',
                'N',
                'final',
                'rel_bias',
                'init',
                'post_em',
                'em_raw',
                'em_winsor',
                'em_trim',
                'floor%',
                'cap%',
                'G_mom',
            ],
            tablefmt='simple',
        )
    )

    if corr_rows:
        print('\nOff-diagonal correlation trace')
        print(
            tabulate(
                corr_rows,
                headers=['dataset', 'part', 'N_corr', 'post_em', 'final_shrunk', 'RMSE'],
                tablefmt='simple',
            )
        )

    print('\nGlobal component bins')
    bin_rows: list[list[str]] = []
    for path in ['full_mom', 'diag_mom', 'component_diag', 'fallback']:
        row = _summarize_bin(f'path={path}', records, lambda r, path=path: r['path'] == path)
        if row is not None:
            bin_rows.append(row)
    for field in ['floor_post_em', 'cap_post_em']:
        for value in [False, True]:
            row = _summarize_bin(f'{field}={value}', records, lambda r, f=field, v=value: r[f] == v)
            if row is not None:
                bin_rows.append(row)

    G_mom_values = np.array([float(r['G_mom']) for r in records])
    if G_mom_values.size:
        edges = np.unique(np.quantile(G_mom_values, [0.0, 0.25, 0.5, 0.75, 1.0]))
        if edges.size > 1:
            for label in sorted({_bin_label(float(v), edges) for v in G_mom_values}):
                row = _summarize_bin(
                    f'G_mom={label}',
                    records,
                    lambda r, label=label, edges=edges: _bin_label(float(r['G_mom']), edges)
                    == label,
                )
                if row is not None:
                    bin_rows.append(row)

    for q_count in sorted({int(r['q']) for r in records}):
        row = _summarize_bin(
            f'q={q_count}', records, lambda r, q_count=q_count: int(r['q']) == q_count
        )
        if row is not None:
            bin_rows.append(row)

    print(tabulate(bin_rows, headers=['bin', 'N', 'final', 'rel_bias', 'RMSE'], tablefmt='simple'))


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-id', default=None, help='Optional single dataset id.')
    parser.add_argument('--partition', default=None, choices=['train', 'valid', 'test'])
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = setup()
    run_srfx_diagnostic(args.data_id, args.partition, args.batch_size)
