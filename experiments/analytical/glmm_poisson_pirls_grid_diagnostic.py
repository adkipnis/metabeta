"""Diagnose Poisson PIRLS full-candidate sigma-grid choices.

This compares the current PIRLS+β prototype against the full-candidate PIRLS sigma grid
on identical rows. It reports which sigma scales are selected and whether the grid is
buying FFX/BLUP gains at the cost of worse sigma estimates.

Example:
    uv run python -u experiments/analytical/glmm_poisson_pirls_grid_diagnostic.py \
        --sizes small medium --max-datasets 1000 --batch-size 32

The default full grid is intentionally conservative: it tests shrinkage and fixed-σ
re-synchronization only (`0.5, 0.75, 1.0`). Larger σ-inflation scales can be supplied
explicitly when diagnosing whether extra FFX gains justify the σ outlier risk.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


SIZES = ['small', 'medium', 'large', 'huge']
PARTITIONS = ['mixed_train', 'sampled_valid', 'sampled_test']
METRICS = ['ffx', 'srfx', 'blup']


def _nrmse(errs: list[np.ndarray], truths: list[np.ndarray]) -> float:
    if not errs:
        return float('nan')
    err = np.concatenate(errs)
    truth = np.concatenate(truths)
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')


def _scaleKey(scale: float, accept: float) -> str:
    if accept < 0.5:
        return 'rejected'
    return f'{scale:.6g}'


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _combos(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    if args.combos:
        return [_parseCombo(combo) for combo in args.combos]

    combos = []
    for size in args.sizes:
        if 'mixed_train' in args.partitions:
            combos.append((f'{size}-p-mixed', 'train', 2))
        if 'sampled_valid' in args.partitions:
            combos.append((f'{size}-p-sampled', 'valid', 0))
        if 'sampled_test' in args.partitions:
            combos.append((f'{size}-p-sampled', 'test', 0))
    return combos


def _parseCombo(combo: str) -> tuple[str, str, int]:
    parts = combo.split(':')
    if len(parts) not in {2, 3}:
        raise ValueError(f'combo must be data_id:partition[:n_epochs], got {combo!r}')
    n_epochs = int(parts[2]) if len(parts) == 3 else 0
    return parts[0], parts[1], n_epochs


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _commonKwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
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


def _fitPirls(
    batch: dict[str, torch.Tensor],
    max_q: int,
    args: argparse.Namespace,
    *,
    sigma_grid: bool,
) -> dict[str, torch.Tensor]:
    return glmm(
        batch['X'],
        batch['y'],
        batch['Z'][..., :max_q],
        batch['mask_n'].float(),
        batch['mask_m'].float(),
        batch['ns'].clamp(min=1).float(),
        batch['n'].float(),
        likelihood_family=2,
        map_refine=True,
        bernoulli_laplace_eb=False,
        normal_laplace_eb=False,
        poisson_laplace_eb='poisson_eb',
        poisson_laplace_eb_diagnostics=True,
        poisson_laplace_pirls_diag=True,
        poisson_laplace_pirls_diag_outer=args.poisson_laplace_pirls_diag_outer,
        poisson_laplace_pirls_diag_inner=args.poisson_laplace_pirls_diag_inner,
        poisson_laplace_pirls_diag_final=args.poisson_laplace_pirls_diag_final,
        poisson_laplace_pirls_diag_damping=args.poisson_laplace_pirls_diag_damping,
        poisson_laplace_pirls_diag_sigma_blend=args.poisson_laplace_pirls_diag_sigma_blend,
        poisson_laplace_pirls_diag_prior_weight=args.poisson_laplace_pirls_diag_prior_weight,
        poisson_marginal_beta=True,
        poisson_marginal_beta_steps=args.poisson_marginal_beta_steps,
        poisson_marginal_beta_damping=args.poisson_marginal_beta_damping,
        poisson_marginal_beta_min_d=args.poisson_marginal_beta_min_d,
        poisson_marginal_beta_max_q=args.poisson_marginal_beta_max_q,
        poisson_marginal_beta_max_step=args.poisson_marginal_beta_max_step,
        poisson_marginal_beta_full_psi_min_q=args.poisson_marginal_beta_full_psi_min_q,
        poisson_laplace_pirls_sigma_grid=sigma_grid,
        poisson_laplace_pirls_sigma_grid_scales=tuple(args.poisson_laplace_pirls_sigma_grid_scales),
        poisson_laplace_pirls_sigma_grid_steps=args.poisson_laplace_pirls_sigma_grid_steps,
        poisson_laplace_pirls_sigma_grid_min_d=args.poisson_laplace_pirls_sigma_grid_min_d,
        poisson_laplace_pirls_sigma_grid_max_q=args.poisson_laplace_pirls_sigma_grid_max_q,
        **_commonKwargs(batch),
    )


def _emptyGroup() -> dict:
    return {
        'n': 0,
        'accepted': [],
        'target_gain': [],
        'sigma_ratio': [],
        'row_delta': {metric: [] for metric in METRICS},
        'errs': {
            method: {metric: [] for metric in METRICS} for method in ['pirls_beta', 'full_grid']
        },
        'truths': {metric: [] for metric in METRICS},
    }


def _addMetric(
    groups: dict[str, dict],
    keys: list[str],
    metric: str,
    base_err: np.ndarray,
    grid_err: np.ndarray,
    truth: np.ndarray,
) -> None:
    base_rmse = _rmse(base_err)
    grid_rmse = _rmse(grid_err)
    for key in keys:
        group = groups[key]
        group['errs']['pirls_beta'][metric].append(base_err)
        group['errs']['full_grid'][metric].append(grid_err)
        group['truths'][metric].append(truth)
        group['row_delta'][metric].append(grid_rmse - base_rmse)


def _recordRow(
    rows: list[dict[str, float | int | str]],
    groups: dict[str, dict],
    data_id: str,
    partition: str,
    dataset_idx: int,
    batch: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    grid: dict[str, torch.Tensor],
    b: int,
    max_q: int,
) -> None:
    d_mask = batch['mask_d'][b].bool()
    q_mask = batch['mask_q'][b, :max_q].bool()
    m_mask = batch['mask_m'][b].bool()
    d = int(d_mask.sum().item())
    q = int(q_mask.sum().item())
    m = int(m_mask.sum().item())
    n = int(batch['n'][b].item())

    beta_true = batch['ffx'][b, d_mask].detach().cpu().numpy()
    sigma_true = batch['sigma_rfx'][b, q_mask].detach().cpu().numpy()
    blup_true = batch['rfx'][b, m_mask][:, q_mask].reshape(-1).detach().cpu().numpy()

    beta_base = base['beta_est'][b, d_mask].detach().cpu().numpy()
    beta_grid = grid['beta_est'][b, d_mask].detach().cpu().numpy()
    sigma_base = base['sigma_rfx_est'][b, q_mask].detach().cpu().numpy()
    sigma_grid = grid['sigma_rfx_est'][b, q_mask].detach().cpu().numpy()
    blup_base = base['blup_est'][b, m_mask][:, q_mask].reshape(-1).detach().cpu().numpy()
    blup_grid = grid['blup_est'][b, m_mask][:, q_mask].reshape(-1).detach().cpu().numpy()

    beta_base_err = beta_base - beta_true
    beta_grid_err = beta_grid - beta_true
    sigma_base_err = sigma_base - sigma_true
    sigma_grid_err = sigma_grid - sigma_true
    blup_base_err = blup_base - blup_true
    blup_grid_err = blup_grid - blup_true

    scale = float(grid.get('poisson_pirls_sigma_grid_scale', torch.ones_like(batch['n']))[b].item())
    accept = float(
        grid.get('poisson_pirls_sigma_grid_accept', torch.zeros_like(batch['n']))[b].item()
    )
    target_fallback = torch.full(
        (batch['X'].shape[0],), float('nan'), device=batch['X'].device, dtype=batch['X'].dtype
    )
    base_target = float(grid.get('poisson_pirls_sigma_grid_base_target', target_fallback)[b].item())
    target = float(grid.get('poisson_pirls_sigma_grid_target', target_fallback)[b].item())
    scale_key = _scaleKey(scale, accept)
    sigma_ratio = float(np.mean(sigma_grid / np.maximum(sigma_base, 1e-8))) if q else float('nan')

    row = {
        'data_id': data_id,
        'partition': partition,
        'dataset_idx': dataset_idx,
        'd': d,
        'q': q,
        'm': m,
        'n': n,
        'accept': accept,
        'scale': scale,
        'target_gain': target - base_target,
        'sigma_ratio': sigma_ratio,
        'ffx_delta': _rmse(beta_grid_err) - _rmse(beta_base_err),
        'srfx_delta': _rmse(sigma_grid_err) - _rmse(sigma_base_err),
        'blup_delta': _rmse(blup_grid_err) - _rmse(blup_base_err),
        'ffx_base_rmse': _rmse(beta_base_err),
        'ffx_grid_rmse': _rmse(beta_grid_err),
        'srfx_base_rmse': _rmse(sigma_base_err),
        'srfx_grid_rmse': _rmse(sigma_grid_err),
        'blup_base_rmse': _rmse(blup_base_err),
        'blup_grid_rmse': _rmse(blup_grid_err),
    }
    rows.append(row)

    keys = [
        'all',
        f'cell={data_id}:{partition}',
        f'scale={scale_key}',
        f'd={_dBin(d)}',
        f'q={_qBin(q)}',
    ]
    if accept >= 0.5:
        keys.append('accepted')
    else:
        keys.append('rejected')
    if row['ffx_delta'] < -1e-8 and row['srfx_delta'] > 1e-8:
        keys.append('ffx_win_sigma_loss')
    elif row['ffx_delta'] < -1e-8 and row['srfx_delta'] <= 1e-8:
        keys.append('ffx_win_sigma_win')
    elif row['ffx_delta'] >= -1e-8 and row['srfx_delta'] > 1e-8:
        keys.append('ffx_flat_sigma_loss')

    for key in keys:
        group = groups[key]
        group['n'] += 1
        group['accepted'].append(accept)
        group['target_gain'].append(row['target_gain'])
        group['sigma_ratio'].append(sigma_ratio)

    _addMetric(groups, keys, 'ffx', beta_base_err, beta_grid_err, beta_true)
    _addMetric(groups, keys, 'srfx', sigma_base_err, sigma_grid_err, sigma_true)
    _addMetric(groups, keys, 'blup', blup_base_err, blup_grid_err, blup_true)


def _dBin(d: int) -> str:
    if d <= 4:
        return '<=4'
    if d <= 8:
        return '5-8'
    if d <= 12:
        return '9-12'
    return '>=13'


def _qBin(q: int) -> str:
    if q <= 1:
        return '1'
    if q <= 2:
        return '2'
    if q <= 4:
        return '3-4'
    return '>=5'


def _summaryRow(label: str, group: dict) -> list[str]:
    n = group['n']
    accepted = float(np.mean(group['accepted'])) if group['accepted'] else float('nan')
    sigma_ratio = float(np.nanmean(group['sigma_ratio'])) if group['sigma_ratio'] else float('nan')
    target_gain = float(np.nanmean(group['target_gain'])) if group['target_gain'] else float('nan')
    return [
        label,
        str(n),
        f'{accepted:.3f}',
        f"{_nrmse(group['errs']['pirls_beta']['ffx'], group['truths']['ffx']):.4f}",
        f"{_nrmse(group['errs']['full_grid']['ffx'], group['truths']['ffx']):.4f}",
        f"{_nrmse(group['errs']['pirls_beta']['srfx'], group['truths']['srfx']):.4f}",
        f"{_nrmse(group['errs']['full_grid']['srfx'], group['truths']['srfx']):.4f}",
        f"{_nrmse(group['errs']['pirls_beta']['blup'], group['truths']['blup']):.4f}",
        f"{_nrmse(group['errs']['full_grid']['blup'], group['truths']['blup']):.4f}",
        f'{sigma_ratio:.3f}',
        f'{target_gain:.3f}',
    ]


def _tradeoffRow(label: str, rows: list[dict[str, float | int | str]]) -> list[str]:
    if not rows:
        return [label, '0'] + ['nan'] * 7
    ffx = np.array([float(row['ffx_delta']) for row in rows])
    srfx = np.array([float(row['srfx_delta']) for row in rows])
    blup = np.array([float(row['blup_delta']) for row in rows])
    ffx_win = ffx < -1e-8
    sigma_loss = srfx > 1e-8
    blup_win = blup < -1e-8
    return [
        label,
        str(len(rows)),
        f'{float(ffx_win.mean()):.3f}',
        f'{float(sigma_loss.mean()):.3f}',
        f'{float((ffx_win & sigma_loss).mean()):.3f}',
        f'{float((ffx_win & ~sigma_loss & blup_win).mean()):.3f}',
        f'{float(np.mean(ffx)):.4f}',
        f'{float(np.mean(srfx)):.4f}',
        f'{float(np.mean(blup)):.4f}',
    ]


def _printSummary(rows: list[dict[str, float | int | str]], groups: dict[str, dict]) -> None:
    print('\nCell summary')
    cell_labels = sorted(label for label in groups if label.startswith('cell='))
    print(
        tabulate(
            [_summaryRow(label.removeprefix('cell='), groups[label]) for label in cell_labels],
            headers=[
                'cell',
                'N',
                'accept',
                'base FFX',
                'grid FFX',
                'base σ',
                'grid σ',
                'base BLUP',
                'grid BLUP',
                'σ ratio',
                'target Δ',
            ],
        )
    )

    print('\nAccepted scale summary')
    scale_labels = sorted(
        (label for label in groups if label.startswith('scale=')),
        key=lambda value: (
            value == 'scale=rejected',
            float('inf') if value == 'scale=rejected' else float(value.removeprefix('scale=')),
        ),
    )
    print(
        tabulate(
            [_summaryRow(label.removeprefix('scale='), groups[label]) for label in scale_labels],
            headers=[
                'scale',
                'N',
                'accept',
                'base FFX',
                'grid FFX',
                'base σ',
                'grid σ',
                'base BLUP',
                'grid BLUP',
                'σ ratio',
                'target Δ',
            ],
        )
    )

    print('\nTradeoffs by cell')
    rows_by_cell: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        rows_by_cell[f'{row["data_id"]}:{row["partition"]}'].append(row)
    print(
        tabulate(
            [_tradeoffRow(label, rows_by_cell[label]) for label in sorted(rows_by_cell)]
            + [_tradeoffRow('all', rows)],
            headers=[
                'cell',
                'N',
                'FFX win',
                'σ loss',
                'FFX win + σ loss',
                'all useful',
                'mean ΔFFX',
                'mean Δσ',
                'mean ΔBLUP',
            ],
        )
    )

    print('\nWorst sigma tradeoffs among FFX-improving rows')
    tradeoff_rows = [
        row for row in rows if float(row['ffx_delta']) < -1e-8 and float(row['srfx_delta']) > 1e-8
    ]
    tradeoff_rows = sorted(tradeoff_rows, key=lambda row: float(row['srfx_delta']), reverse=True)
    print(
        tabulate(
            [
                [
                    f'{row["data_id"]}:{row["partition"]}',
                    row['dataset_idx'],
                    row['d'],
                    row['q'],
                    f'{float(row["scale"]):.6g}',
                    f'{float(row["ffx_delta"]):.4f}',
                    f'{float(row["srfx_delta"]):.4f}',
                    f'{float(row["blup_delta"]):.4f}',
                    f'{float(row["sigma_ratio"]):.3f}',
                    f'{float(row["target_gain"]):.3f}',
                ]
                for row in tradeoff_rows[:12]
            ],
            headers=[
                'cell',
                'idx',
                'd',
                'q',
                'scale',
                'ΔFFX',
                'Δσ',
                'ΔBLUP',
                'σ ratio',
                'target Δ',
            ],
        )
    )


def _writeCsv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def runDiagnostic(args: argparse.Namespace) -> None:
    rows: list[dict[str, float | int | str]] = []
    groups: dict[str, dict] = defaultdict(_emptyGroup)

    with torch.no_grad():
        for data_id, partition, n_epochs in _combos(args):
            cfg = loadDataConfig(data_id)
            max_q = int(cfg['max_q'])
            n_seen = 0
            print(f'Running {data_id}:{partition}', flush=True)
            for path in _paths(data_id, partition, n_epochs):
                for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                    if args.max_datasets is not None:
                        remaining = args.max_datasets - n_seen
                        if remaining <= 0:
                            break
                        if batch['X'].shape[0] > remaining:
                            batch = _sliceBatch(batch, remaining)
                    batch = toDevice(batch, torch.device('cpu'))
                    base = _fitPirls(batch, max_q, args, sigma_grid=False)
                    grid = _fitPirls(batch, max_q, args, sigma_grid=True)
                    B = batch['X'].shape[0]
                    for b in range(B):
                        _recordRow(
                            rows,
                            groups,
                            data_id,
                            partition,
                            n_seen + b,
                            batch,
                            base,
                            grid,
                            b,
                            max_q,
                        )
                    n_seen += B
                if args.max_datasets is not None and n_seen >= args.max_datasets:
                    break

    _printSummary(rows, groups)
    if args.save_csv is not None:
        _writeCsv(args.save_csv, rows)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=['small', 'medium'], choices=SIZES)
    parser.add_argument('--partitions', nargs='+', default=PARTITIONS, choices=PARTITIONS)
    parser.add_argument('--combos', nargs='+', default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-datasets', type=int, default=1000)
    parser.add_argument('--save-csv', type=Path, default=None)
    parser.add_argument('--poisson-marginal-beta-steps', type=int, default=4)
    parser.add_argument('--poisson-marginal-beta-damping', type=float, default=0.7)
    parser.add_argument('--poisson-marginal-beta-min-d', type=int, default=1)
    parser.add_argument('--poisson-marginal-beta-max-q', type=int, default=None)
    parser.add_argument('--poisson-marginal-beta-max-step', type=float, default=1.0)
    parser.add_argument('--poisson-marginal-beta-full-psi-min-q', type=int, default=3)
    parser.add_argument('--poisson-laplace-pirls-diag-outer', type=int, default=4)
    parser.add_argument('--poisson-laplace-pirls-diag-inner', type=int, default=1)
    parser.add_argument('--poisson-laplace-pirls-diag-final', type=int, default=2)
    parser.add_argument('--poisson-laplace-pirls-diag-damping', type=float, default=0.5)
    parser.add_argument('--poisson-laplace-pirls-diag-sigma-blend', type=float, default=0.5)
    parser.add_argument('--poisson-laplace-pirls-diag-prior-weight', type=float, default=4.0)
    parser.add_argument('--poisson-laplace-pirls-sigma-grid-scales', type=float, nargs='+', default=[0.5, 0.75, 1.0])
    parser.add_argument('--poisson-laplace-pirls-sigma-grid-steps', type=int, default=2)
    parser.add_argument('--poisson-laplace-pirls-sigma-grid-min-d', type=int, default=1)
    parser.add_argument('--poisson-laplace-pirls-sigma-grid-max-q', type=int, default=None)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    runDiagnostic(setup())
