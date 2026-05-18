"""Diagnose where fast normal GLMM estimates lag diagonal R-INLA.

This is intentionally small-sample: R-INLA is seconds per dataset, so the script
is for failure-mode attribution, not the required benchmark table.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from experiments.analytical.glmm_inla_comparison import _flatten, _inla_estimate
from metabeta.analytical.glmm import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


DATA_IDS = ['small-n-mixed', 'medium-n-mixed', 'large-n-mixed', 'huge-n-mixed']
METHODS = ['normal_eb']
TAIL_METRICS = [
    'ffx_eb_rmse',
    'sigma_eb_rmse',
    'blup_eb_rmse',
    'beta_abs_err_max',
    'cap_ffx_eb_rmse',
]


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')


def _safeCond(x: np.ndarray) -> float:
    if x.shape[0] == 0 or x.shape[1] == 0:
        return float('nan')
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return float('inf')
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float('inf')
    s_max = float(np.max(s))
    s_min = float(np.min(s))
    if s_min <= 1e-12:
        return float('inf')
    return s_max / s_min


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float('nan')
    x_sel = x[finite]
    y_sel = y[finite]
    if float(np.std(x_sel)) <= 1e-12 or float(np.std(y_sel)) <= 1e-12:
        return float('nan')
    return float(np.corrcoef(x_sel, y_sel)[0, 1])


def _nanStat(x: np.ndarray, fn) -> float:
    finite = x[np.isfinite(x)]
    return float(fn(finite)) if finite.size else float('nan')


def _binByD(d: int) -> str:
    if d <= 4:
        return 'd<=4'
    if d <= 8:
        return '5<=d<=8'
    if d <= 12:
        return '9<=d<=12'
    return 'd>=13'


def _binByM(m: int) -> str:
    if m <= 20:
        return 'm<=20'
    if m <= 60:
        return '21<=m<=60'
    return 'm>=61'


def _binByCond(value: float) -> str:
    if not np.isfinite(value):
        return 'cond=inf'
    if value < 10:
        return 'cond<10'
    if value < 100:
        return '10<=cond<100'
    if value < 1000:
        return '100<=cond<1000'
    return 'cond>=1000'


def _binByR2(value: float) -> str:
    if not np.isfinite(value):
        return 'r2=nan'
    if value < 0.25:
        return 'r2<0.25'
    if value < 0.5:
        return '0.25<=r2<0.5'
    if value < 0.75:
        return '0.5<=r2<0.75'
    return 'r2>=0.75'


def _methodKwargs(method: str) -> dict[str, bool]:
    if method == 'normal_eb':
        return {'map_refine': True, 'bernoulli_laplace_eb': False, 'normal_laplace_eb': True}
    raise ValueError(f'unsupported analytical method: {method}')


def _pathsForArgs(cfg: dict, partition: str, n_epochs: int) -> list[Path]:
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _normalCommon(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    return {
        'eta_rfx': batch.get('eta_rfx'),
        'mask_q': batch.get('mask_q'),
        'nu_ffx': batch.get('nu_ffx'),
        'tau_ffx': batch.get('tau_ffx'),
        'family_ffx': batch.get('family_ffx'),
        'tau_rfx': batch.get('tau_rfx'),
        'family_sigma_rfx': batch.get('family_sigma_rfx'),
        'tau_eps': batch.get('tau_eps'),
        'family_sigma_eps': batch.get('family_sigma_eps'),
        'mask_d': batch.get('mask_d'),
    }


def _normalStats(batch: dict[str, torch.Tensor], Zm: torch.Tensor) -> dict[str, dict]:
    return {
        method: glmm(
            batch['X'],
            batch['y'],
            Zm,
            batch['mask_n'].float(),
            batch['mask_m'].float(),
            batch['ns'].clamp(min=1).float(),
            batch['n'].float(),
            likelihood_family=0,
            **_methodKwargs(method),
            **_normalCommon(batch),
        )
        for method in METHODS
    }


def _designDiagnostics(ds_flat: dict) -> dict[str, float | np.ndarray]:
    x = np.asarray(ds_flat['X'], dtype=float)
    z = np.asarray(ds_flat['Z'], dtype=float)
    groups = np.asarray(ds_flat['groups'], dtype=int)
    d = x.shape[1]
    m = int(ds_flat['m'])

    resid_parts = []
    for g in range(m):
        sel = groups == g
        if not np.any(sel):
            continue
        x_g = x[sel]
        z_g = z[sel]
        if z_g.shape[1] == 0:
            resid_parts.append(x_g)
            continue
        try:
            coef = np.linalg.lstsq(z_g, x_g, rcond=1e-8)[0]
            resid_parts.append(x_g - z_g @ coef)
        except np.linalg.LinAlgError:
            resid_parts.append(x_g)

    rx = np.vstack(resid_parts) if resid_parts else x
    x_slope = x[:, 1:] if d > 1 else x[:, :0]
    rx_slope = rx[:, 1:] if d > 1 else rx[:, :0]
    raw_ss = np.sum(np.square(x), axis=0)
    resid_ss = np.sum(np.square(rx), axis=0)
    fe_re_r2 = np.where(raw_ss > 1e-12, 1.0 - resid_ss / raw_ss, np.nan)
    fe_re_r2 = np.clip(fe_re_r2, 0.0, 1.0)
    slope_r2 = fe_re_r2[1:] if d > 1 else np.array([], dtype=float)

    return {
        'x_cond': _safeCond(x_slope),
        'rx_cond': _safeCond(rx_slope),
        'max_fe_re_r2': _nanStat(fe_re_r2, np.max),
        'max_slope_fe_re_r2': _nanStat(slope_r2, np.max),
        'mean_slope_fe_re_r2': _nanStat(slope_r2, np.mean),
        'fe_re_r2': fe_re_r2,
    }


def _candidateFromBatch(
    data_id: str,
    dataset_idx: int,
    batch: dict[str, torch.Tensor],
    stats: dict[str, dict],
    b: int,
    active_d: np.ndarray,
    active_q: np.ndarray,
) -> dict:
    ffx_true = batch['ffx'].cpu().numpy()
    srfx_true = batch['sigma_rfx'].cpu().numpy()
    rfx_true = batch['rfx'].cpu().numpy()
    ns_arr = batch['ns'].cpu().numpy()
    m = int(batch['m'][b].item())
    n = int(batch['n'][b].item())
    flat = _flatten(batch, b, active_d, active_q)
    design = _designDiagnostics(flat)
    ns_b = ns_arr[b, :m].astype(float)
    ns_mean = float(np.mean(ns_b)) if ns_b.size else float('nan')
    ns_cv = float(np.std(ns_b) / max(ns_mean, 1e-8)) if ns_b.size else float('nan')

    method = METHODS[0]
    beta_est = stats[method]['beta_est'][b, active_d].detach().cpu().numpy()
    sigma_est = stats[method]['sigma_rfx_est'][b, active_q].detach().cpu().numpy()
    blup_est = stats[method]['blup_est'][b, :m][:, active_q].detach().cpu().numpy()
    beta_err = beta_est - ffx_true[b, active_d]
    sigma_err = sigma_est - srfx_true[b, active_q]
    blup_err = (blup_est - rfx_true[b, :m][:, active_q]).reshape(-1)
    cap_default = torch.zeros(batch['X'].shape[0], dtype=torch.float32)
    cap_hit = float(
        stats[method].get('normal_map_beta_prior_capped', cap_default)[b].detach().cpu().item()
    )
    beta_abs_err = np.abs(beta_err)
    fe_re_r2 = design['fe_re_r2']
    diag = {
        'data_id': data_id,
        'dataset_idx': dataset_idx,
        'method': method,
        'd': len(active_d),
        'q': len(active_q),
        'm': m,
        'n': n,
        'ns_min': float(np.min(ns_b)) if ns_b.size else float('nan'),
        'ns_max': float(np.max(ns_b)) if ns_b.size else float('nan'),
        'ns_cv': ns_cv,
        'x_cond': design['x_cond'],
        'rx_cond': design['rx_cond'],
        'max_fe_re_r2': design['max_fe_re_r2'],
        'max_slope_fe_re_r2': design['max_slope_fe_re_r2'],
        'mean_slope_fe_re_r2': design['mean_slope_fe_re_r2'],
        'beta_r2_corr': _corr(beta_abs_err, fe_re_r2[: len(active_d)]),
        'beta_prior_cap': cap_hit,
        'beta_abs_err_mean': float(np.mean(beta_abs_err)),
        'beta_abs_err_max': float(np.max(beta_abs_err)),
        'sigma_true_mean': float(np.mean(srfx_true[b, active_q])),
        'sigma_eb_mean': float(np.mean(sigma_est)),
        'sigma_eb_bias_mean': float(np.mean(sigma_err)),
        'sigma_eps_true': float(batch['sigma_eps'][b].cpu().item()),
        'sigma_eps_eb': float(stats[method]['sigma_eps_est'][b, 0].detach().cpu().item()),
        'ffx_eb_rmse': _rmse(beta_err),
        'sigma_eb_rmse': _rmse(sigma_err),
        'blup_eb_rmse': _rmse(blup_err),
        'ffx_inla_rmse': float('nan'),
        'sigma_inla_rmse': float('nan'),
        'blup_inla_rmse': float('nan'),
        'ffx_gap': float('nan'),
        'sigma_gap': float('nan'),
        'blup_gap': float('nan'),
        'sigma_inla_mean': float('nan'),
        'sigma_inla_bias_mean': float('nan'),
    }
    return {
        'diag': diag,
        'flat': flat,
        'ffx_true': ffx_true[b, active_d],
        'srfx_true': srfx_true[b, active_q],
        'rfx_true': rfx_true[b, :m][:, active_q],
    }


def _candidateScore(candidate: dict, metric: str) -> float:
    diag = candidate['diag']
    if metric == 'cap_ffx_eb_rmse':
        return diag['ffx_eb_rmse'] + 1000.0 * diag['beta_prior_cap']
    return diag[metric]


def _appendInlaRows(rows: list[dict], diag: dict) -> None:
    for metric, value in [
        ('ffx', diag['ffx_eb_rmse']),
        ('sigma', diag['sigma_eb_rmse']),
        ('blup', diag['blup_eb_rmse']),
    ]:
        rows.append(
            {
                'data_id': diag['data_id'],
                'method': diag['method'],
                'metric': metric,
                'method_rmse': value,
                'inla_rmse': diag[f'{metric}_inla_rmse'],
                'd_bin': _binByD(diag['d']),
                'm_bin': _binByM(diag['m']),
                'rx_cond_bin': _binByCond(float(diag['rx_cond'])),
                'slope_r2_bin': _binByR2(float(diag['max_slope_fe_re_r2'])),
                'cap_bin': 'cap' if diag['beta_prior_cap'] else 'no_cap',
                'q': diag['q'],
            }
        )


def runTailDiagnostic(args: argparse.Namespace) -> None:
    rows = []
    gap_rows = []
    t0 = time.perf_counter()
    scanned_total = 0
    for data_id in args.data_ids:
        cfg = loadDataConfig(data_id)
        max_q = cfg['max_q']
        candidates = []
        seen = 0
        for path in _pathsForArgs(cfg, args.partition, args.n_epochs):
            if seen >= args.tail_scan:
                break
            for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                if seen >= args.tail_scan:
                    break
                if seen + batch['X'].shape[0] > args.tail_scan:
                    batch = _sliceBatch(batch, args.tail_scan - seen)
                batch = toDevice(batch, torch.device('cpu'))
                B = batch['X'].shape[0]
                stats = _normalStats(batch, batch['Z'][..., :max_q])
                mask_d = batch['mask_d'].cpu().numpy().astype(bool)
                mask_q = batch['mask_q'].cpu().numpy().astype(bool)
                for b in range(B):
                    active_d = np.flatnonzero(mask_d[b])
                    active_q = np.flatnonzero(mask_q[b])
                    candidates.append(
                        _candidateFromBatch(data_id, seen, batch, stats, b, active_d, active_q)
                    )
                    seen += 1
        scanned_total += seen
        selected = sorted(
            candidates,
            key=lambda candidate: _candidateScore(candidate, args.tail_metric),
            reverse=True,
        )[: args.tail_k]
        for candidate in selected:
            diag = candidate['diag']
            est = _inla_estimate(
                candidate['flat'],
                likelihood_family=0,
                re_correlation='diagonal',
            )
            if est is None:
                continue
            m = diag['m']
            diag['ffx_inla_rmse'] = _rmse(est['beta'] - candidate['ffx_true'])
            diag['sigma_inla_rmse'] = _rmse(est['sigma_rfx'] - candidate['srfx_true'])
            diag['blup_inla_rmse'] = _rmse((est['blups'][:m] - candidate['rfx_true']).reshape(-1))
            diag['ffx_gap'] = diag['ffx_eb_rmse'] - diag['ffx_inla_rmse']
            diag['sigma_gap'] = diag['sigma_eb_rmse'] - diag['sigma_inla_rmse']
            diag['blup_gap'] = diag['blup_eb_rmse'] - diag['blup_inla_rmse']
            diag['sigma_inla_mean'] = float(np.mean(est['sigma_rfx']))
            diag['sigma_inla_bias_mean'] = float(np.mean(est['sigma_rfx'] - candidate['srfx_true']))
            gap_rows.append(diag)
            _appendInlaRows(rows, diag)

    print(
        f'scanned {scanned_total} rows, ran INLA on {len(gap_rows)} selected tail rows '
        f'in {time.perf_counter() - t0:.1f}s'
    )
    _printSummary(rows, 'data_id')
    _printSummary(rows, 'd_bin')
    _printSummary(rows, 'rx_cond_bin')
    _printSummary(rows, 'slope_r2_bin')
    _printSummary(rows, 'cap_bin')
    _printRanked(gap_rows, 'ffx_gap', args.top_k)
    _printRanked(gap_rows, 'sigma_gap', args.top_k)
    _printRanked(gap_rows, 'blup_gap', args.top_k)
    if args.output_csv:
        _writeCsv(gap_rows, Path(args.output_csv))


def runDiagnostic(args: argparse.Namespace) -> None:
    rows = []
    gap_rows = []
    t0 = time.perf_counter()
    for data_id in args.data_ids:
        cfg = loadDataConfig(data_id)
        max_q = cfg['max_q']
        paths = (
            [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, args.n_epochs + 1)]
            if args.partition == 'train'
            else [dataFilePath(cfg['data_id'], args.partition)]
        )
        seen = 0
        for path in paths:
            if seen >= args.n_inla:
                break
            for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                if seen >= args.n_inla:
                    break
                if seen + batch['X'].shape[0] > args.n_inla:
                    batch = _sliceBatch(batch, args.n_inla - seen)
                batch = toDevice(batch, torch.device('cpu'))
                B = batch['X'].shape[0]
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
                stats = {
                    method: glmm(
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['ns'].clamp(min=1).float(),
                        batch['n'].float(),
                        likelihood_family=0,
                        **_methodKwargs(method),
                        **common,
                    )
                    for method in METHODS
                }
                ffx_true = batch['ffx'].cpu().numpy()
                srfx_true = batch['sigma_rfx'].cpu().numpy()
                rfx_true = batch['rfx'].cpu().numpy()
                mask_d = batch['mask_d'].cpu().numpy().astype(bool)
                mask_q = batch['mask_q'].cpu().numpy().astype(bool)
                m_arr = batch['m'].cpu().numpy()
                n_arr = batch['n'].cpu().numpy()
                ns_arr = batch['ns'].cpu().numpy()
                sigma_eps_true = batch['sigma_eps'].cpu().numpy()

                for b in range(B):
                    active_d = np.flatnonzero(mask_d[b])
                    active_q = np.flatnonzero(mask_q[b])
                    m = int(m_arr[b])
                    n = int(n_arr[b])
                    flat = _flatten(batch, b, active_d, active_q)
                    design = _designDiagnostics(flat)
                    est = _inla_estimate(
                        flat,
                        likelihood_family=0,
                        re_correlation='diagonal',
                    )
                    seen += 1
                    if est is None:
                        continue
                    inla_err = {
                        'ffx': _rmse(est['beta'] - ffx_true[b, active_d]),
                        'sigma': _rmse(est['sigma_rfx'] - srfx_true[b, active_q]),
                        'blup': _rmse(
                            (est['blups'][:m] - rfx_true[b, :m][:, active_q]).reshape(-1)
                        ),
                    }
                    ns_b = ns_arr[b, :m].astype(float)
                    ns_mean = float(np.mean(ns_b)) if ns_b.size else float('nan')
                    ns_cv = float(np.std(ns_b) / max(ns_mean, 1e-8)) if ns_b.size else float('nan')
                    for method in METHODS:
                        beta_est = stats[method]['beta_est'][b, active_d].detach().cpu().numpy()
                        sigma_est = (
                            stats[method]['sigma_rfx_est'][b, active_q].detach().cpu().numpy()
                        )
                        blup_est = (
                            stats[method]['blup_est'][b, :m][:, active_q].detach().cpu().numpy()
                        )
                        beta_err = beta_est - ffx_true[b, active_d]
                        sigma_err = sigma_est - srfx_true[b, active_q]
                        blup_err = (blup_est - rfx_true[b, :m][:, active_q]).reshape(-1)
                        method_err = {
                            'ffx': _rmse(beta_err),
                            'sigma': _rmse(sigma_err),
                            'blup': _rmse(blup_err),
                        }
                        cap_default = torch.zeros(B, dtype=torch.float32)
                        cap_hit = float(
                            stats[method]
                            .get('normal_map_beta_prior_capped', cap_default)[b]
                            .detach()
                            .cpu()
                            .item()
                        )
                        beta_abs_err = np.abs(beta_err)
                        fe_re_r2 = design['fe_re_r2']
                        beta_r2_corr = _corr(beta_abs_err, fe_re_r2[: len(active_d)])
                        diag = {
                            'data_id': data_id,
                            'dataset_idx': seen - 1,
                            'method': method,
                            'd': len(active_d),
                            'q': len(active_q),
                            'm': m,
                            'n': n,
                            'ns_min': float(np.min(ns_b)) if ns_b.size else float('nan'),
                            'ns_max': float(np.max(ns_b)) if ns_b.size else float('nan'),
                            'ns_cv': ns_cv,
                            'x_cond': design['x_cond'],
                            'rx_cond': design['rx_cond'],
                            'max_fe_re_r2': design['max_fe_re_r2'],
                            'max_slope_fe_re_r2': design['max_slope_fe_re_r2'],
                            'mean_slope_fe_re_r2': design['mean_slope_fe_re_r2'],
                            'beta_r2_corr': beta_r2_corr,
                            'beta_prior_cap': cap_hit,
                            'beta_abs_err_mean': float(np.mean(beta_abs_err)),
                            'beta_abs_err_max': float(np.max(beta_abs_err)),
                            'sigma_true_mean': float(np.mean(srfx_true[b, active_q])),
                            'sigma_eb_mean': float(np.mean(sigma_est)),
                            'sigma_inla_mean': float(np.mean(est['sigma_rfx'])),
                            'sigma_eb_bias_mean': float(np.mean(sigma_err)),
                            'sigma_inla_bias_mean': float(
                                np.mean(est['sigma_rfx'] - srfx_true[b, active_q])
                            ),
                            'sigma_eps_true': float(sigma_eps_true[b]),
                            'sigma_eps_eb': float(
                                stats[method]['sigma_eps_est'][b, 0].detach().cpu().item()
                            ),
                            'ffx_eb_rmse': method_err['ffx'],
                            'ffx_inla_rmse': inla_err['ffx'],
                            'ffx_gap': method_err['ffx'] - inla_err['ffx'],
                            'sigma_eb_rmse': method_err['sigma'],
                            'sigma_inla_rmse': inla_err['sigma'],
                            'sigma_gap': method_err['sigma'] - inla_err['sigma'],
                            'blup_eb_rmse': method_err['blup'],
                            'blup_inla_rmse': inla_err['blup'],
                            'blup_gap': method_err['blup'] - inla_err['blup'],
                        }
                        gap_rows.append(diag)
                        for metric, value in method_err.items():
                            rows.append(
                                {
                                    'data_id': data_id,
                                    'method': method,
                                    'metric': metric,
                                    'method_rmse': value,
                                    'inla_rmse': inla_err[metric],
                                    'd_bin': _binByD(len(active_d)),
                                    'm_bin': _binByM(m),
                                    'rx_cond_bin': _binByCond(float(design['rx_cond'])),
                                    'slope_r2_bin': _binByR2(float(design['max_slope_fe_re_r2'])),
                                    'cap_bin': 'cap' if cap_hit else 'no_cap',
                                    'q': len(active_q),
                                }
                            )

    print(f'completed {len(rows)} method/metric rows in {time.perf_counter() - t0:.1f}s')
    _printSummary(rows, 'data_id')
    _printSummary(rows, 'd_bin')
    _printSummary(rows, 'm_bin')
    _printSummary(rows, 'rx_cond_bin')
    _printSummary(rows, 'slope_r2_bin')
    _printSummary(rows, 'cap_bin')
    _printRanked(gap_rows, 'ffx_gap', args.top_k)
    _printRanked(gap_rows, 'sigma_gap', args.top_k)
    _printRanked(gap_rows, 'blup_gap', args.top_k)
    if args.output_csv:
        _writeCsv(gap_rows, Path(args.output_csv))


def _printSummary(rows: list[dict], by: str) -> None:
    table = []
    keys = sorted({row[by] for row in rows})
    for key in keys:
        for method in METHODS:
            for metric in ['ffx', 'sigma', 'blup']:
                selected = [
                    row
                    for row in rows
                    if row[by] == key and row['method'] == method and row['metric'] == metric
                ]
                if not selected:
                    continue
                method_rmse = np.array([row['method_rmse'] for row in selected])
                inla_rmse = np.array([row['inla_rmse'] for row in selected])
                delta = method_rmse - inla_rmse
                table.append(
                    [
                        key,
                        method,
                        metric,
                        len(selected),
                        f'{float(method_rmse.mean()):.4f}',
                        f'{float(inla_rmse.mean()):.4f}',
                        f'{float(delta.mean()):+.4f}',
                        f'{float(np.mean(delta > 0.0)):.2f}',
                    ]
                )
    print(f'\nGrouped by {by}')
    print(
        tabulate(
            table,
            headers=['bin', 'method', 'metric', 'N', 'method RMSE', 'INLA RMSE', 'Δ', 'worse%'],
            tablefmt='github',
        )
    )


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return 'inf' if value > 0 else 'nan'
    return f'{value:.4f}'


def _printRanked(rows: list[dict], metric: str, top_k: int) -> None:
    selected = sorted(rows, key=lambda row: row[metric], reverse=True)[:top_k]
    table = []
    for row in selected:
        table.append(
            [
                row['data_id'],
                row['dataset_idx'],
                _fmt(row[metric]),
                _fmt(row['ffx_eb_rmse']),
                _fmt(row['ffx_inla_rmse']),
                _fmt(row['sigma_eb_rmse']),
                _fmt(row['sigma_inla_rmse']),
                _fmt(row['blup_eb_rmse']),
                _fmt(row['blup_inla_rmse']),
                row['d'],
                row['q'],
                row['m'],
                _fmt(row['ns_cv']),
                _fmt(row['rx_cond']),
                _fmt(row['max_slope_fe_re_r2']),
                int(row['beta_prior_cap']),
                _fmt(row['sigma_true_mean']),
                _fmt(row['sigma_eb_mean']),
                _fmt(row['sigma_inla_mean']),
            ]
        )
    print(f'\nTop {len(selected)} by {metric} (positive means Normal EB worse than INLA)')
    print(
        tabulate(
            table,
            headers=[
                'data',
                'idx',
                'gap',
                'EB FFX',
                'INLA FFX',
                'EB sigma',
                'INLA sigma',
                'EB BLUP',
                'INLA BLUP',
                'd',
                'q',
                'm',
                'ns_cv',
                'rx_cond',
                'max RE R2',
                'cap',
                'sig true',
                'sig EB',
                'sig INLA',
            ],
            tablefmt='github',
        )
    )


def _writeCsv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwrote per-dataset diagnostics to {path}')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-ids',   nargs='+', default=DATA_IDS)
    parser.add_argument('--partition',            default='train')
    parser.add_argument('--n-epochs',   type=int,  default=2)
    parser.add_argument('--n-inla',     type=int,  default=10)
    parser.add_argument('--batch-size', type=int,  default=16)
    parser.add_argument('--top-k',      type=int,  default=10)
    parser.add_argument('--output-csv',           default='')
    parser.add_argument('--tail-scan',  type=int,  default=0,
                        help='scan this many rows per dataset analytically before tail INLA')
    parser.add_argument('--tail-k',     type=int,  default=12,
                        help='run INLA on this many selected tail rows per dataset')
    parser.add_argument('--tail-metric',          choices=TAIL_METRICS, default='cap_ffx_eb_rmse')
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    args = setup()
    if args.tail_scan > 0:
        runTailDiagnostic(args)
    else:
        runDiagnostic(args)
