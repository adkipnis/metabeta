"""Experiment: analytical GLMM vs R-INLA on Bernoulli/Normal GLMM datasets.

Compares analytical GLMM summaries against R-INLA on Bernoulli or Normal datasets.
The analytical side can run the raw estimator and the current EB-refined estimator.

RE prior: PC prior P(σ_j > τ_rfx_j) = 0.317 per dimension for uncorrelated
datasets (eta_rfx=0, or q=1).  For correlated datasets (eta_rfx>0, q=2):
iid2d model with Wishart prior W(q+1, V) where V_{jj}=1/((q+1)*τ_rfx_j²),
plus a copy term for the second dimension.
FE prior: N(ν_ffx_j, τ_ffx_j²) via control.fixed.
Normal family residual prior: PC prior P(σ_eps > τ_eps) = 0.317.

Usage (from repo root):
    uv run python -u experiments/analytical/glmm_inla_comparison.py
    uv run python -u experiments/analytical/glmm_inla_comparison.py \\
        --data-ids small-b-sampled,small-n-sampled --n-inla 100
    uv run python -u experiments/analytical/glmm_inla_comparison.py \\
        --data-ids small-n-sampled --n-inla 200 --partition test \\
        --analytical-methods raw,current
    uv run python -u experiments/analytical/glmm_inla_comparison.py \\
        --data-ids large-p-sampled --partition valid \\
        --n-inla 1000 --n-total 1000 --analytical-methods raw,current \\
        --save-inla-rows-dir experiments/analytical/inla_runs/poisson_rows
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


ANALYTICAL_METHODS = ('raw', 'current', 'normal_eb', 'normal_beta_grid')
INLA_ROW_OUTPUT_DIR = 'experiments/analytical/inla_runs/row_estimates'


def _inlaPath(data_path: Path) -> Path:
    return data_path.with_suffix('.inla.npz')


def _checkInlaFits(paths: list[Path], data_id: str) -> None:
    missing = [_inlaPath(p) for p in paths if not _inlaPath(p).exists()]
    if not missing:
        return
    parts = data_id.split('-')
    size, fam = parts[0], parts[1]
    ds_type = '-'.join(parts[2:])
    fam_int = {'n': 0, 'b': 1, 'p': 2}.get(fam, '?')
    lines = [
        f'Missing INLA fit file(s) for {data_id}:',
        *(f'  {p}' for p in missing),
        '',
        'Generate with:',
        f'  uv run python -m metabeta.simulation.fit --method inla \\',
        f'      --size {size} --family {fam_int} --ds_type {ds_type} \\',
        f'      --partition <partition> [--epoch <epoch>] --idx <idx>',
        '  # then reintegrate:',
        f'  uv run python -m metabeta.simulation.fit --method inla \\',
        f'      --size {size} --family {fam_int} --ds_type {ds_type} \\',
        f'      --partition <partition> [--epoch <epoch>] --reintegrate',
    ]
    print('ERROR: ' + '\n'.join(lines), file=sys.stderr)
    sys.exit(1)


def _loadInlaFits(path: Path) -> dict[str, np.ndarray]:
    with np.load(_inlaPath(path)) as f:
        out = {
            'beta': f['inla_ffx'].copy(),
            'sigma_rfx': f['inla_sigma_rfx'].copy(),
            'blups': f['inla_rfx'].copy(),
            'wall_s': f['inla_wall_s'].copy(),
            'failed': f['inla_failed'].copy().astype(bool),
        }
        if 'inla_sigma_rfx_mode' in f:
            out['sigma_rfx_mode'] = f['inla_sigma_rfx_mode'].copy()
        return out


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.std(truth))
    return float(np.sqrt(np.mean(err**2))) / max(denom, 1e-8)


def _bias(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def _parseAnalyticalMethods(value: str) -> list[str]:
    methods = [method.strip().lower() for method in value.split(',') if method.strip()]
    invalid = sorted(set(methods) - set(ANALYTICAL_METHODS))
    if invalid:
        raise ValueError(f'unsupported analytical method(s): {", ".join(invalid)}')
    return methods or ['raw', 'current']


def _methodLabel(method: str, likelihood_family: int) -> str:
    if method == 'raw':
        return 'RAW'
    if method in {'current', 'normal_eb'} and likelihood_family == 1:
        return 'BERNOULLI-EB'
    if method == 'current' and likelihood_family == 0:
        return 'NORMAL-EB-TAIL-BETA'
    if method == 'normal_eb' and likelihood_family == 0:
        return 'NORMAL-EB'
    if method == 'normal_beta_grid' and likelihood_family == 0:
        return 'NORMAL-EB-BETA-GRID'
    if method == 'normal_eb':
        return 'NORMAL-EB'
    if method == 'current' and likelihood_family == 2:
        return 'POISSON-EB-SIGMA-GRID'
    return method.upper()


def _metricLists() -> dict[str, list[np.ndarray] | list[float]]:
    return {
        'be': [],
        'bt': [],
        'se': [],
        'st': [],
        're': [],
        'rt': [],
        'wall': [],
    }


def _flat(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values) if values else np.array([np.nan])


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _pad1d(values: np.ndarray, size: int, fill: float = np.nan) -> np.ndarray:
    out = np.full(size, fill, dtype=float)
    n = min(size, len(values))
    if n > 0:
        out[:n] = values[:n]
    return out


def _pad2d(values: np.ndarray, rows: int, cols: int, fill: float = np.nan) -> np.ndarray:
    out = np.full((rows, cols), fill, dtype=float)
    r = min(rows, values.shape[0])
    c = min(cols, values.shape[1]) if values.ndim == 2 else 0
    if r > 0 and c > 0:
        out[:r, :c] = values[:r, :c]
    return out


def _rowScalar(stats: dict, key: str, b: int) -> float:
    value = stats.get(key)
    if value is None or not torch.is_tensor(value) or value.ndim == 0 or b >= value.shape[0]:
        return float('nan')
    return float(value[b].detach().cpu().item())


def _rowCountStats(batch: dict[str, torch.Tensor], b: int) -> dict[str, float]:
    m = int(batch['m'][b].item())
    y = batch['y'][b, :m].detach().cpu().numpy()
    mask = batch['mask_n'][b, :m].detach().cpu().numpy().astype(bool)
    y_active = y[mask]
    if not y_active.size:
        return {
            'y_mean': float('nan'),
            'y_var': float('nan'),
            'y_max': float('nan'),
            'y_zero_frac': float('nan'),
            'y_tail_frac': float('nan'),
        }
    mean = float(np.mean(y_active))
    std = float(np.std(y_active))
    tail_cut = mean + 2.0 * std
    return {
        'y_mean': mean,
        'y_var': float(np.var(y_active)),
        'y_max': float(np.max(y_active)),
        'y_zero_frac': float(np.mean(y_active == 0.0)),
        'y_tail_frac': float(np.mean(y_active > tail_cut)) if std > 0.0 else 0.0,
    }


def _batchVectorOrNan(
    batch: dict[str, torch.Tensor],
    key: str,
    b: int,
    size: int,
) -> np.ndarray:
    value = batch.get(key)
    if value is None or not torch.is_tensor(value):
        return np.full(size, np.nan, dtype=float)
    return value[b, :size].detach().cpu().numpy().astype(float)


def _appendInlaRowRecord(
    records: list[dict],
    *,
    data_id: str,
    partition: str,
    dataset_idx: int,
    batch: dict[str, torch.Tensor],
    b: int,
    active_d: np.ndarray,
    active_q: np.ndarray,
    stats_np: dict[str, dict],
    stats_torch: dict[str, dict],
    analytical_methods: list[str],
    est: dict,
    inla_wall_s: float,
    max_d: int,
    max_q: int,
) -> None:
    m_b = int(batch['m'][b].item())
    max_m = batch['rfx'].shape[1]
    record = {
        'data_id': data_id,
        'partition': partition,
        'dataset_idx': dataset_idx,
        'n': int(batch['n'][b].item()),
        'm': m_b,
        'd': int(active_d.size),
        'q': int(active_q.size),
        'mask_d': _pad1d(np.isin(np.arange(max_d), active_d).astype(float), max_d, fill=0.0),
        'mask_q': _pad1d(np.isin(np.arange(max_q), active_q).astype(float), max_q, fill=0.0),
        'beta_true': batch['ffx'][b, :max_d].detach().cpu().numpy().astype(float),
        'sigma_rfx_true': batch['sigma_rfx'][b, :max_q].detach().cpu().numpy().astype(float),
        'blup_true': batch['rfx'][b, :max_m, :max_q].detach().cpu().numpy().astype(float),
        'nu_ffx': _batchVectorOrNan(batch, 'nu_ffx', b, max_d),
        'tau_ffx': _batchVectorOrNan(batch, 'tau_ffx', b, max_d),
        'tau_rfx': _batchVectorOrNan(batch, 'tau_rfx', b, max_q),
        'eta_rfx': (
            float(batch['eta_rfx'][b].detach().cpu().item())
            if torch.is_tensor(batch.get('eta_rfx'))
            else float('nan')
        ),
        'beta_inla': _pad1d(est['beta'], max_d),
        'sigma_rfx_inla': _pad1d(est['sigma_rfx'], max_q),
        'sigma_rfx_mode_inla': _pad1d(
            est.get('sigma_rfx_mode', np.full(len(active_q), np.nan)), max_q
        ),
        'blup_inla': _pad2d(est['blups'], max_m, max_q),
        'inla_wall_s': float(inla_wall_s),
        **_rowCountStats(batch, b),
    }
    for method in analytical_methods:
        record[f'beta_{method}'] = stats_np[method]['beta'][b, :max_d].astype(float)
        record[f'sigma_rfx_{method}'] = stats_np[method]['srfx'][b, :max_q].astype(float)
        record[f'blup_{method}'] = stats_np[method]['blup'][b, :max_m, :max_q].astype(float)
        record[f'wall_ms_{method}'] = float(stats_np[method]['wall'] * 1000.0)
        if 'map_srfx' in stats_np[method]:
            record[f'sigma_rfx_map_{method}'] = stats_np[method]['map_srfx'][b, :max_q].astype(
                float
            )

    current_stats = stats_torch.get('current')
    if current_stats is not None:
        for key in (
            'laplace_eb_accept',
            'laplace_eb_sigma_prior_capped',
            'laplace_eb_blup_fallback',
            'laplace_eb_beta_jump',
            'poisson_marginal_beta_gate',
            'poisson_marginal_beta_accept',
            'poisson_marginal_beta_jump',
            'poisson_pirls_sigma_grid_gate',
            'poisson_pirls_sigma_grid_accept',
            'poisson_pirls_sigma_grid_scale',
        ):
            record[key] = _rowScalar(current_stats, key, b)
    records.append(record)


def _saveInlaRowRecords(
    records: list[dict],
    output_dir: Path,
    *,
    data_id: str,
    partition: str,
    analytical_methods: list[str],
    likelihood_family: int,
) -> None:
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{data_id}_{partition}_inla_rows.npz'
    keys = sorted(records[0].keys())
    payload = {
        'data_id': np.array(data_id),
        'partition': np.array(partition),
        'likelihood_family': np.array(likelihood_family, dtype=np.int64),
        'analytical_methods': np.array(analytical_methods),
    }
    for key in keys:
        values = [row[key] for row in records]
        if isinstance(values[0], str):
            payload[key] = np.array(values)
        else:
            payload[key] = np.stack(values) if np.ndim(values[0]) > 0 else np.array(values)
    np.savez_compressed(path, **payload)
    print(f'Wrote INLA row estimates: {path} ({len(records)} rows)', flush=True)


def _breakdown_srfx(err: np.ndarray, truth: np.ndarray, label: str) -> str:
    """Bias/RMSE breakdown by true σ_rfx in four quantile bins."""
    finite = np.isfinite(truth) & np.isfinite(err)
    edges = np.nanpercentile(truth[finite], [0, 25, 50, 75, 100])
    edges = np.unique(np.round(edges, 3))
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = finite & (truth >= lo) & (truth <= (hi if i == len(edges) - 2 else hi - 1e-9))
        if sel.sum() < 2:
            continue
        e = err[sel]
        rows.append(
            [
                f'{lo:.3f}–{hi:.3f}',
                int(sel.sum()),
                f'{float(np.nanmean(e)):+.4f}',
                f'{float(np.sqrt(np.mean(e**2))):.4f}',
            ]
        )
    return tabulate(rows, headers=[f'σ_rfx_true ({label})', 'N', 'Bias', 'RMSE'], tablefmt='simple')


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def run_one_dataset(
    data_id: str,
    partition: str = 'test',
    n_total: int = 0,
    n_epochs: int = 1,
    analytical_methods: list[str] | None = None,
    save_inla_rows_dir: Path | None = None,
    device: torch.device | None = None,
    skip_analytical: bool = False,
) -> dict:
    """Run analytical GLMM methods vs R-INLA on one dataset, return metrics dict."""
    if device is None:
        device = torch.device('cpu')
    if analytical_methods is None:
        analytical_methods = ['raw', 'current']
    if skip_analytical:
        analytical_methods = []
    data_cfg = loadDataConfig(data_id)
    max_d = data_cfg['max_d']
    max_q = data_cfg['max_q']
    likelihood_family = data_cfg.get('likelihood_family', 0)

    if partition == 'train':
        paths = [dataFilePath(data_cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
        paths = [p for p in paths if p.exists()]
    elif partition == 'test':
        paths = [dataFilePath(data_cfg['data_id'], 'test')]
    else:
        paths = [dataFilePath(data_cfg['data_id'], 'valid')]

    assert paths and paths[0].exists(), f'No data at {paths[0] if paths else data_id}/{partition}'

    _checkInlaFits(paths, data_id)

    all_metrics = {method: _metricLists() for method in analytical_methods}
    matched_metrics = {method: _metricLists() for method in analytical_methods}
    inla_metrics = _metricLists()
    inla_n_ok = 0
    inla_row_records: list[dict] = []

    ds_bar = tqdm(
        total=n_total if n_total > 0 else None,
        desc=f'{data_id}/{partition} datasets',
        unit='ds',
        file=sys.stderr,
        leave=False,
    )

    with torch.no_grad():
        n_seen = 0
        done = False
        for path in paths:
            if done:
                break
            inla_data = _loadInlaFits(path)
            dl = Dataloader(path, batch_size=32, sortish=False, shuffle=False)
            file_n_seen = 0
            for batch in dl:
                if n_total > 0 and n_seen >= n_total:
                    done = True
                    break
                if n_total > 0:
                    remaining = n_total - n_seen
                    if batch['X'].shape[0] > remaining:
                        batch = _sliceBatch(batch, remaining)
                batch = toDevice(batch, device)
                B = batch['X'].shape[0]
                Zm = batch['Z'][..., :max_q]

                stats_np = {}
                stats_torch = {}
                for method in analytical_methods:
                    if method == 'raw':
                        method_kwargs = {
                            'map_refine': False,
                            'bernoulli_laplace_eb': False,
                            'normal_laplace_eb': False,
                            'poisson_laplace_eb': False,
                        }
                    elif method == 'normal_eb' and likelihood_family == 0:
                        method_kwargs = {
                            'map_refine': True,
                            'bernoulli_laplace_eb': False,
                            'normal_laplace_eb': True,
                            'normal_laplace_eb_sigma_grid_refine': False,
                            'normal_beta_sigma_grid': False,
                            'normal_beta_tail_grid': False,
                        }
                    elif method == 'normal_beta_grid' and likelihood_family == 0:
                        method_kwargs = {
                            'map_refine': True,
                            'bernoulli_laplace_eb': False,
                            'normal_laplace_eb': True,
                            'normal_laplace_eb_sigma_grid_refine': False,
                            'normal_beta_sigma_grid': True,
                            'normal_beta_sigma_grid_scales': (0.75, 1.0, 1.3333333),
                            'normal_beta_tail_grid': False,
                        }
                    else:
                        method_kwargs = {'map_refine': True}
                    if (
                        save_inla_rows_dir is not None
                        and method == 'current'
                        and likelihood_family == 2
                    ):
                        method_kwargs['poisson_laplace_eb_diagnostics'] = True
                    t0 = time.perf_counter()
                    stats = glmm(
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['ns'].clamp(min=1).float(),
                        batch['n'].float(),
                        likelihood_family=likelihood_family,
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
                        **method_kwargs,
                    )
                    elapsed = time.perf_counter() - t0
                    stats_torch[method] = stats
                    snp: dict = {
                        'beta': stats['beta_est'].cpu().numpy(),
                        'srfx': stats['sigma_rfx_est'].cpu().numpy(),
                        'blup': stats['blup_est'].cpu().numpy(),
                        'wall': elapsed / B,
                    }
                    if 'normal_map_sigma_rfx' in stats:
                        snp['map_srfx'] = stats['normal_map_sigma_rfx'].cpu().numpy()
                    stats_np[method] = snp

                ffx_true = batch['ffx'].cpu().numpy()
                srfx_true = batch['sigma_rfx'].cpu().numpy()
                rfx_true = batch['rfx'].cpu().numpy()
                mask_d_np = (
                    batch['mask_d'].cpu().numpy().astype(bool)
                    if 'mask_d' in batch
                    else np.ones((B, max_d), dtype=bool)
                )
                mask_q_np = (
                    batch['mask_q'].cpu().numpy().astype(bool)
                    if 'mask_q' in batch
                    else np.ones((B, max_q), dtype=bool)
                )
                m_np = batch['m'].cpu().numpy()

                for b in range(B):
                    active_d = np.flatnonzero(mask_d_np[b])
                    active_q = np.flatnonzero(mask_q_np[b])
                    m_b = int(m_np[b])
                    re_t = rfx_true[b, :m_b][:, active_q].reshape(-1)

                    per_method_errors = {}
                    for method in analytical_methods:
                        be = stats_np[method]['beta'][b, active_d] - ffx_true[b, active_d]
                        se = stats_np[method]['srfx'][b, active_q] - srfx_true[b, active_q]
                        re_e = (
                            stats_np[method]['blup'][b, :m_b][:, active_q]
                            - rfx_true[b, :m_b][:, active_q]
                        ).reshape(-1)
                        per_method_errors[method] = {'be': be, 'se': se, 're': re_e}
                        store = all_metrics[method]
                        store['be'].append(be)
                        store['bt'].append(ffx_true[b, active_d])
                        store['se'].append(se)
                        store['st'].append(srfx_true[b, active_q])
                        store['re'].append(re_e)
                        store['rt'].append(re_t)
                        store['wall'].append(stats_np[method]['wall'])

                    inla_idx = file_n_seen
                    n_seen += 1
                    file_n_seen += 1
                    ds_bar.update(1)

                    if inla_data['failed'][inla_idx]:
                        continue

                    inla_elapsed = float(inla_data['wall_s'][inla_idx])
                    est = {
                        'beta': inla_data['beta'][inla_idx, active_d],
                        'sigma_rfx': inla_data['sigma_rfx'][inla_idx, active_q],
                        'blups': inla_data['blups'][inla_idx, :m_b, :][:, active_q],
                    }
                    if 'sigma_rfx_mode' in inla_data:
                        est['sigma_rfx_mode'] = inla_data['sigma_rfx_mode'][inla_idx, active_q]
                    inla_metrics['wall'].append(inla_elapsed)
                    inla_n_ok += 1
                    for method in analytical_methods:
                        store = matched_metrics[method]
                        store['be'].append(per_method_errors[method]['be'])
                        store['bt'].append(ffx_true[b, active_d])
                        store['se'].append(per_method_errors[method]['se'])
                        store['st'].append(srfx_true[b, active_q])
                        store['re'].append(per_method_errors[method]['re'])
                        store['rt'].append(re_t)
                    inla_metrics['be'].append(est['beta'] - ffx_true[b, active_d])
                    inla_metrics['bt'].append(ffx_true[b, active_d])
                    inla_metrics['se'].append(est['sigma_rfx'] - srfx_true[b, active_q])
                    inla_metrics['st'].append(srfx_true[b, active_q])
                    inla_metrics['re'].append(
                        (est['blups'][:m_b] - rfx_true[b, :m_b][:, active_q]).reshape(-1)
                    )
                    inla_metrics['rt'].append(re_t)
                    if save_inla_rows_dir is not None:
                        _appendInlaRowRecord(
                            inla_row_records,
                            data_id=data_id,
                            partition=partition,
                            dataset_idx=n_seen - 1,
                            batch=batch,
                            b=b,
                            active_d=active_d,
                            active_q=active_q,
                            stats_np=stats_np,
                            stats_torch=stats_torch,
                            analytical_methods=analytical_methods,
                            est=est,
                            inla_wall_s=inla_elapsed,
                            max_d=max_d,
                            max_q=max_q,
                        )

    ds_bar.close()
    if save_inla_rows_dir is not None:
        _saveInlaRowRecords(
            inla_row_records,
            save_inla_rows_dir,
            data_id=data_id,
            partition=partition,
            analytical_methods=analytical_methods,
            likelihood_family=likelihood_family,
        )
    all_flat = {
        method: {key: _flat(values) for key, values in store.items() if key != 'wall'}
        for method, store in all_metrics.items()
    }
    matched_flat = {
        method: {key: _flat(values) for key, values in store.items() if key != 'wall'}
        for method, store in matched_metrics.items()
    }
    inla = {key: _flat(values) for key, values in inla_metrics.items() if key != 'wall'}

    n_all = len(all_metrics[analytical_methods[0]]['be']) if analytical_methods else n_seen
    metrics = {
        'data_id': data_id,
        'n_analytical': n_all,
        'n_inla': inla_n_ok,
        'analytical_methods': analytical_methods,
        'inla_ffx': _nrmse(inla['be'], inla['bt']) if inla_n_ok > 0 else float('nan'),
        'inla_srfx': _nrmse(inla['se'], inla['st']) if inla_n_ok > 0 else float('nan'),
        'inla_blup': _nrmse(inla['re'], inla['rt']) if inla_n_ok > 0 else float('nan'),
    }
    for method in analytical_methods:
        vals = all_flat[method]
        metrics[f'{method}_ffx'] = _nrmse(vals['be'], vals['bt'])
        metrics[f'{method}_srfx'] = _nrmse(vals['se'], vals['st'])
        metrics[f'{method}_blup'] = _nrmse(vals['re'], vals['rt'])
        method_wall = np.array(all_metrics[method]['wall'])
        metrics[f'{method}_wall_ms'] = float(method_wall.mean() * 1000.0)
    inla_wall = np.array(inla_metrics['wall'])
    metrics['inla_wall_s'] = float(inla_wall.mean()) if inla_wall.size else float('nan')

    sep = '=' * 70
    print(sep)
    print(f'  {data_id}  |  partition={partition}  N={n_all}  family={likelihood_family}')
    print(sep)

    family_label = {0: 'Normal', 1: 'Bernoulli', 2: 'Poisson'}.get(
        likelihood_family, str(likelihood_family)
    )
    print(f'Family: {family_label}')

    for method in analytical_methods:
        label = _methodLabel(method, likelihood_family)
        vals = all_flat[method]
        print(f'\n{label} — all q (N={n_all})')
        print(
            tabulate(
                [
                    ['FFX (β)', f'{metrics[f"{method}_ffx"]:.4f}', f'{_bias(vals["be"]):+.4f}'],
                    ['σ_rfx', f'{metrics[f"{method}_srfx"]:.4f}', f'{_bias(vals["se"]):+.4f}'],
                    ['BLUP', f'{metrics[f"{method}_blup"]:.4f}', f'{_bias(vals["re"]):+.4f}'],
                ],
                headers=['Parameter', 'NRMSE', 'Bias'],
                tablefmt='simple',
            )
        )

    if inla_n_ok > 0:
        n_matched = (
            len(matched_metrics[analytical_methods[0]]['be']) if analytical_methods else inla_n_ok
        )
        print(
            f'\nR-INLA — matched (N={n_matched}, INLA ok={inla_n_ok})'
            if not analytical_methods
            else f'\nAnalytical vs R-INLA — matched (N={n_matched}, INLA ok={inla_n_ok})'
        )
        rows = []
        for method in analytical_methods:
            label = _methodLabel(method, likelihood_family)
            vals = matched_flat[method]
            rows.extend(
                [
                    [
                        label,
                        'FFX (β)',
                        f'{_nrmse(vals["be"], vals["bt"]):.4f}',
                        f'{_bias(vals["be"]):+.4f}',
                    ],
                    [
                        label,
                        'σ_rfx',
                        f'{_nrmse(vals["se"], vals["st"]):.4f}',
                        f'{_bias(vals["se"]):+.4f}',
                    ],
                    [
                        label,
                        'BLUP',
                        f'{_nrmse(vals["re"], vals["rt"]):.4f}',
                        f'{_bias(vals["re"]):+.4f}',
                    ],
                ]
            )
        rows.extend(
            [
                ['R-INLA', 'FFX (β)', f'{metrics["inla_ffx"]:.4f}', f'{_bias(inla["be"]):+.4f}'],
                ['R-INLA', 'σ_rfx', f'{metrics["inla_srfx"]:.4f}', f'{_bias(inla["se"]):+.4f}'],
                ['R-INLA', 'BLUP', f'{metrics["inla_blup"]:.4f}', f'{_bias(inla["re"]):+.4f}'],
            ]
        )
        print(tabulate(rows, headers=['Method', 'Parameter', 'NRMSE', 'Bias'], tablefmt='simple'))

    for method in analytical_methods:
        label = _methodLabel(method, likelihood_family)
        vals = all_flat[method]
        print(f'\nσ_rfx bias by true σ_rfx bin — {label}')
        print(_breakdown_srfx(vals['se'], vals['st'], label))
    if inla_n_ok > 0:
        print('\nσ_rfx bias by true σ_rfx bin — R-INLA')
        print(_breakdown_srfx(inla['se'], inla['st'], 'R-INLA'))

    print()
    for method in analytical_methods:
        label = _methodLabel(method, likelihood_family)
        w = np.array(all_metrics[method]['wall'])
        print(
            f'Wall time — {label}: mean={w.mean()*1000:.2f} ms  '
            f'median={np.median(w)*1000:.2f} ms/ds'
        )
    if inla_metrics['wall']:
        w = np.array(inla_metrics['wall'])
        print(f'Wall time — INLA: mean={w.mean():.3f} s  median={np.median(w):.3f} s/ds')
    print()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_ids: list[str],
    partition: str = 'test',
    n_epochs: int = 1,
    n_total: int = 0,
    analytical_methods: list[str] | None = None,
    save_inla_rows_dir: str = INLA_ROW_OUTPUT_DIR,
    skip_analytical: bool = False,
) -> None:
    if analytical_methods is None:
        analytical_methods = ['raw', 'current']
    if skip_analytical:
        print('Analytical methods: (skipped)')
    else:
        print(f'Analytical methods: {", ".join(analytical_methods)}')
    if n_total > 0:
        print(f'n_total={n_total}')
    row_output_dir = Path(save_inla_rows_dir) if save_inla_rows_dir else None
    if row_output_dir is not None:
        print(f'INLA row estimates: {row_output_dir}')
    print()

    all_metrics = []
    for data_id in data_ids:
        all_metrics.append(
            run_one_dataset(
                data_id=data_id,
                partition=partition,
                n_total=n_total,
                n_epochs=n_epochs,
                analytical_methods=analytical_methods,
                save_inla_rows_dir=row_output_dir,
                skip_analytical=skip_analytical,
            )
        )

    if len(all_metrics) > 1:
        print('=' * 70)
        print('  SUMMARY — NRMSE across datasets  (F=FFX, S=σ_rfx, B=BLUP)')
        print('=' * 70)
        rows = []
        for m in all_metrics:

            def fmt(v: float) -> str:
                return f'{v:.4f}' if not np.isnan(v) else '—'

            rows.append(
                [m['data_id'], m['n_analytical']]
                + [
                    fmt(m[f'{method}_{suffix}'])
                    for method in analytical_methods
                    for suffix in ['ffx', 'srfx', 'blup']
                ]
                + [fmt(m['inla_ffx']), fmt(m['inla_srfx']), fmt(m['inla_blup'])]
            )
        method_headers = [
            f'{method.upper()}-{short}'
            for method in analytical_methods
            for short in ['F', 'S', 'B']
        ]
        print(
            tabulate(
                rows,
                headers=['Dataset', 'N'] + method_headers + ['INLA-F', 'INLA-S', 'INLA-B'],
                tablefmt='simple',
            )
        )
        time_rows = []
        for m in all_metrics:
            time_rows.append(
                [m['data_id']]
                + [f'{m[f"{method}_wall_ms"]:.2f}' for method in analytical_methods]
                + [fmt(m['inla_wall_s'])]
            )
        print()
        print('  SUMMARY — wall time per dataset')
        print(
            tabulate(
                time_rows,
                headers=['Dataset']
                + [f'{method.upper()} ms' for method in analytical_methods]
                + ['INLA s'],
                tablefmt='simple',
            )
        )


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='Analytical GLMM vs R-INLA comparison')
    parser.add_argument('--data-ids',  default='small-b-sampled',
                        help='comma-separated data config ids (default: small-b-sampled)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs',  default=1,   type=int)
    parser.add_argument('--n-total',   default=0,   type=int, help='cap total datasets per data_id (0=all)')
    parser.add_argument('--analytical-methods', default='raw,current',
                        help='comma-separated analytical methods: raw,current,normal_eb,normal_beta_grid')
    parser.add_argument('--save-inla-rows-dir', default=INLA_ROW_OUTPUT_DIR,
                        help='directory for compressed per-row INLA estimate .npz files')
    parser.add_argument('--no-save-inla-rows', action='store_true',
                        help='disable per-row INLA estimate saving')
    parser.add_argument('--no-analytical', action='store_true',
                        help='skip all glmm() calls; only evaluate precomputed INLA fits')
    # fmt: on
    a = parser.parse_args()
    main(
        data_ids=[x.strip() for x in a.data_ids.split(',')],
        partition=a.partition,
        n_epochs=a.n_epochs,
        n_total=a.n_total,
        analytical_methods=_parseAnalyticalMethods(a.analytical_methods),
        save_inla_rows_dir='' if a.no_save_inla_rows else a.save_inla_rows_dir,
        skip_analytical=a.no_analytical,
    )
