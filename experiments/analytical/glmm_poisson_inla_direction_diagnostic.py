"""Diagnose whether Poisson EB moves RAW estimates toward R-INLA.

This script is intentionally small-sample because each R-INLA fit costs seconds.
It answers whether current Poisson EB mostly improves the true-error metric, moves
the estimates toward INLA, or changes sigma without correcting fixed effects.

Example:
    uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
        --data-ids small-p-mixed small-p-sampled --partition train --n-inla 20
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from experiments.analytical.glmm_inla_comparison import (
    _checkInlaFits,
    _loadInlaFits,
    _rowCountStats,
    _rowScalar,
    _sliceBatch,
)
from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


DATA_IDS = ['small-p-mixed', 'small-p-sampled']


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 1:
        return float('nan')
    x_sel = x[finite]
    y_sel = y[finite]
    denom = float(np.linalg.norm(x_sel) * np.linalg.norm(y_sel))
    if denom <= 1e-12:
        return float('nan')
    return float(np.dot(x_sel, y_sel) / denom)


def _gain(raw_dist: float, current_dist: float) -> float:
    if not np.isfinite(raw_dist) or raw_dist <= 1e-12:
        return float('nan')
    return float((raw_dist - current_dist) / raw_dist)


def _binByD(d: int) -> str:
    if d <= 4:
        return 'd<=4'
    if d <= 8:
        return '5<=d<=8'
    if d <= 12:
        return '9<=d<=12'
    return 'd>=13'


def _binByQ(q: int) -> str:
    if q <= 2:
        return 'q<=2'
    if q <= 4:
        return '3<=q<=4'
    return 'q>=5'


def _binByZeroFrac(value: float) -> str:
    if not np.isfinite(value):
        return 'zero=nan'
    if value < 0.1:
        return 'zero<0.1'
    if value < 0.3:
        return '0.1<=zero<0.3'
    if value < 0.6:
        return '0.3<=zero<0.6'
    return 'zero>=0.6'


def _binByMeanY(value: float) -> str:
    if not np.isfinite(value):
        return 'mean_y=nan'
    if value < 0.5:
        return 'mean_y<0.5'
    if value < 1.0:
        return '0.5<=mean_y<1'
    if value < 2.0:
        return '1<=mean_y<2'
    return 'mean_y>=2'


def _pathsForArgs(cfg: dict, partition: str, n_epochs: int) -> list[Path]:
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


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


def _fitPoissonMethods(batch: dict[str, torch.Tensor], max_q: int) -> dict[str, dict]:
    raw = glmm(
        batch['X'],
        batch['y'],
        batch['Z'][..., :max_q],
        batch['mask_n'].float(),
        batch['mask_m'].float(),
        batch['ns'].clamp(min=1).float(),
        batch['n'].float(),
        likelihood_family=2,
        map_refine=False,
        bernoulli_laplace_eb=False,
        normal_laplace_eb=False,
        poisson_laplace_eb=False,
        **_commonKwargs(batch),
    )
    current = glmm(
        batch['X'],
        batch['y'],
        batch['Z'][..., :max_q],
        batch['mask_n'].float(),
        batch['mask_m'].float(),
        batch['ns'].clamp(min=1).float(),
        batch['n'].float(),
        likelihood_family=2,
        map_refine=True,
        poisson_laplace_eb_diagnostics=True,
        **_commonKwargs(batch),
    )
    return {'raw': raw, 'current': current}


def _activeMasks(
    batch: dict[str, torch.Tensor], max_d: int, max_q: int
) -> tuple[np.ndarray, np.ndarray]:
    B = batch['X'].shape[0]
    mask_d = (
        batch['mask_d'].cpu().numpy().astype(bool)
        if 'mask_d' in batch
        else np.ones((B, max_d), dtype=bool)
    )
    mask_q = (
        batch['mask_q'].cpu().numpy().astype(bool)
        if 'mask_q' in batch
        else np.ones((B, max_q), dtype=bool)
    )
    return mask_d, mask_q


def _extractRow(
    data_id: str,
    partition: str,
    dataset_idx: int,
    batch: dict[str, torch.Tensor],
    stats: dict[str, dict],
    b: int,
    active_d: np.ndarray,
    active_q: np.ndarray,
    est: dict,
) -> dict[str, float | int | str]:
    m = int(batch['m'][b].item())
    n = int(batch['n'][b].item())
    count_stats = _rowCountStats(batch, b)

    beta_true = batch['ffx'][b, active_d].detach().cpu().numpy()
    sigma_true = batch['sigma_rfx'][b, active_q].detach().cpu().numpy()
    rfx_true = batch['rfx'][b, :m][:, active_q].detach().cpu().numpy()

    beta_raw = stats['raw']['beta_est'][b, active_d].detach().cpu().numpy()
    beta_current = stats['current']['beta_est'][b, active_d].detach().cpu().numpy()
    sigma_raw = stats['raw']['sigma_rfx_est'][b, active_q].detach().cpu().numpy()
    sigma_current = stats['current']['sigma_rfx_est'][b, active_q].detach().cpu().numpy()
    blup_raw = stats['raw']['blup_est'][b, :m][:, active_q].detach().cpu().numpy()
    blup_current = stats['current']['blup_est'][b, :m][:, active_q].detach().cpu().numpy()

    beta_inla = est['beta']
    sigma_inla = est['sigma_rfx']
    blup_inla = est['blups']

    raw_ffx = _rmse(beta_raw - beta_true)
    current_ffx = _rmse(beta_current - beta_true)
    inla_ffx = _rmse(beta_inla - beta_true)
    raw_sigma = _rmse(sigma_raw - sigma_true)
    current_sigma = _rmse(sigma_current - sigma_true)
    inla_sigma = _rmse(sigma_inla - sigma_true)
    raw_blup = _rmse((blup_raw - rfx_true).reshape(-1))
    current_blup = _rmse((blup_current - rfx_true).reshape(-1))
    inla_blup = _rmse((blup_inla - rfx_true).reshape(-1))

    raw_to_inla_ffx = _rmse(beta_raw - beta_inla)
    current_to_inla_ffx = _rmse(beta_current - beta_inla)
    raw_to_inla_sigma = _rmse(sigma_raw - sigma_inla)
    current_to_inla_sigma = _rmse(sigma_current - sigma_inla)

    current_stats = stats['current']

    return {
        'data_id': data_id,
        'partition': partition,
        'dataset_idx': dataset_idx,
        'd': int(active_d.size),
        'q': int(active_q.size),
        'm': m,
        'n': n,
        **count_stats,
        'accept': _rowScalar(current_stats, 'laplace_eb_accept', b),
        'sigma_capped': _rowScalar(current_stats, 'laplace_eb_sigma_prior_capped', b),
        'blup_fallback': _rowScalar(current_stats, 'laplace_eb_blup_fallback', b),
        'beta_jump': _rowScalar(current_stats, 'laplace_eb_beta_jump', b),
        'marginal_beta_gate': _rowScalar(current_stats, 'poisson_marginal_beta_gate', b),
        'marginal_beta_accept': _rowScalar(current_stats, 'poisson_marginal_beta_accept', b),
        'marginal_beta_jump': _rowScalar(current_stats, 'poisson_marginal_beta_jump', b),
        'sigma_grid_gate': _rowScalar(current_stats, 'poisson_sigma_grid_gate', b),
        'sigma_grid_accept': _rowScalar(current_stats, 'poisson_sigma_grid_accept', b),
        'sigma_grid_scale': _rowScalar(current_stats, 'poisson_sigma_grid_scale', b),
        'raw_ffx_rmse': raw_ffx,
        'current_ffx_rmse': current_ffx,
        'inla_ffx_rmse': inla_ffx,
        'raw_sigma_rmse': raw_sigma,
        'current_sigma_rmse': current_sigma,
        'inla_sigma_rmse': inla_sigma,
        'raw_blup_rmse': raw_blup,
        'current_blup_rmse': current_blup,
        'inla_blup_rmse': inla_blup,
        'raw_to_inla_ffx': raw_to_inla_ffx,
        'current_to_inla_ffx': current_to_inla_ffx,
        'raw_to_inla_sigma': raw_to_inla_sigma,
        'current_to_inla_sigma': current_to_inla_sigma,
        'ffx_true_gain': raw_ffx - current_ffx,
        'sigma_true_gain': raw_sigma - current_sigma,
        'blup_true_gain': raw_blup - current_blup,
        'ffx_inla_gap': current_ffx - inla_ffx,
        'sigma_inla_gap': current_sigma - inla_sigma,
        'blup_inla_gap': current_blup - inla_blup,
        'ffx_distance_gain': _gain(raw_to_inla_ffx, current_to_inla_ffx),
        'sigma_distance_gain': _gain(raw_to_inla_sigma, current_to_inla_sigma),
        'ffx_shift_cos': _cosine(beta_current - beta_raw, beta_inla - beta_raw),
        'sigma_shift_cos': _cosine(sigma_current - sigma_raw, sigma_inla - sigma_raw),
        'sigma_true_mean': float(np.mean(sigma_true)) if sigma_true.size else float('nan'),
        'sigma_raw_mean': float(np.mean(sigma_raw)) if sigma_raw.size else float('nan'),
        'sigma_current_mean': float(np.mean(sigma_current)) if sigma_current.size else float('nan'),
        'sigma_inla_mean': float(np.mean(sigma_inla)) if sigma_inla.size else float('nan'),
    }


def _mean(rows: list[dict], key: str) -> float:
    values = np.array([row[key] for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float('nan')


def _median(rows: list[dict], key: str) -> float:
    values = np.array([row[key] for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float('nan')


def _frac(rows: list[dict], key: str, threshold: float = 0.0) -> float:
    values = np.array([row[key] for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite > threshold)) if finite.size else float('nan')


def _summaryKey(row: dict, by: str) -> str:
    if by == 'd_bin':
        return _binByD(int(row['d']))
    if by == 'q_bin':
        return _binByQ(int(row['q']))
    if by == 'zero_bin':
        return _binByZeroFrac(float(row['y_zero_frac']))
    if by == 'mean_y_bin':
        return _binByMeanY(float(row['y_mean']))
    if by == 'sigma_grid_gate':
        val = row['sigma_grid_gate']
        return 'sg_gate=nan' if not np.isfinite(val) else ('sg_gate=1' if val > 0.5 else 'sg_gate=0')
    if by == 'marginal_beta_gate':
        val = row['marginal_beta_gate']
        return 'mb_gate=nan' if not np.isfinite(val) else ('mb_gate=1' if val > 0.5 else 'mb_gate=0')
    return str(row[by])


def _printOverall(rows: list[dict]) -> None:
    table = [
        [
            len(rows),
            f'{_mean(rows, "raw_ffx_rmse"):.4f}',
            f'{_mean(rows, "current_ffx_rmse"):.4f}',
            f'{_mean(rows, "inla_ffx_rmse"):.4f}',
            f'{_mean(rows, "ffx_true_gain"):+.4f}',
            f'{_median(rows, "ffx_true_gain"):+.4f}',
            f'{_frac(rows, "ffx_true_gain"):.2f}',
            f'{_mean(rows, "ffx_distance_gain"):+.4f}',
            f'{_median(rows, "ffx_distance_gain"):+.4f}',
            f'{_frac(rows, "ffx_distance_gain"):.2f}',
            f'{_mean(rows, "ffx_shift_cos"):+.4f}',
        ]
    ]
    print('\nFFX direction summary')
    print(
        tabulate(
            table,
            headers=[
                'N',
                'RAW RMSE',
                'EB RMSE',
                'INLA RMSE',
                'true gain',
                'med true gain',
                'gain%',
                'INLA dist gain',
                'med dist gain',
                'toward%',
                'shift cos',
            ],
            tablefmt='github',
        )
    )

    table = [
        [
            len(rows),
            f'{_mean(rows, "raw_sigma_rmse"):.4f}',
            f'{_mean(rows, "current_sigma_rmse"):.4f}',
            f'{_mean(rows, "inla_sigma_rmse"):.4f}',
            f'{_mean(rows, "sigma_true_gain"):+.4f}',
            f'{_median(rows, "sigma_true_gain"):+.4f}',
            f'{_frac(rows, "sigma_true_gain"):.2f}',
            f'{_mean(rows, "sigma_distance_gain"):+.4f}',
            f'{_median(rows, "sigma_distance_gain"):+.4f}',
            f'{_frac(rows, "sigma_distance_gain"):.2f}',
            f'{_mean(rows, "sigma_shift_cos"):+.4f}',
            f'{_mean(rows, "accept"):.3f}',
            f'{_mean(rows, "sigma_capped"):.3f}',
        ]
    ]
    print('\nSigma direction summary')
    print(
        tabulate(
            table,
            headers=[
                'N',
                'RAW RMSE',
                'EB RMSE',
                'INLA RMSE',
                'true gain',
                'med true gain',
                'gain%',
                'INLA dist gain',
                'med dist gain',
                'toward%',
                'shift cos',
                'accept',
                'cap',
            ],
            tablefmt='github',
        )
    )

    table = [
        [
            len(rows),
            f'{_mean(rows, "raw_blup_rmse"):.4f}',
            f'{_mean(rows, "current_blup_rmse"):.4f}',
            f'{_mean(rows, "inla_blup_rmse"):.4f}',
            f'{_mean(rows, "blup_true_gain"):+.4f}',
            f'{_median(rows, "blup_true_gain"):+.4f}',
            f'{_frac(rows, "blup_true_gain"):.2f}',
            f'{_mean(rows, "blup_inla_gap"):+.4f}',
        ]
    ]
    print('\nBLUP direction summary')
    print(
        tabulate(
            table,
            headers=['N', 'RAW RMSE', 'EB RMSE', 'INLA RMSE', 'true gain', 'med gain', 'gain%', 'INLA gap'],
            tablefmt='github',
        )
    )


def _printGrouped(rows: list[dict], by: str) -> None:
    table = []
    for key in sorted({_summaryKey(row, by) for row in rows}):
        selected = [row for row in rows if _summaryKey(row, by) == key]
        table.append(
            [
                key,
                len(selected),
                f'{_mean(selected, "ffx_true_gain"):+.4f}',
                f'{_mean(selected, "ffx_inla_gap"):+.4f}',
                f'{_mean(selected, "ffx_distance_gain"):+.4f}',
                f'{_median(selected, "ffx_distance_gain"):+.4f}',
                f'{_mean(selected, "ffx_shift_cos"):+.4f}',
                f'{_mean(selected, "sigma_true_gain"):+.4f}',
                f'{_mean(selected, "sigma_inla_gap"):+.4f}',
                f'{_mean(selected, "sigma_distance_gain"):+.4f}',
                f'{_median(selected, "sigma_distance_gain"):+.4f}',
                f'{_mean(selected, "blup_inla_gap"):+.4f}',
                f'{_mean(selected, "accept"):.3f}',
                f'{_mean(selected, "sigma_capped"):.3f}',
            ]
        )
    print(f'\nGrouped by {by}')
    print(
        tabulate(
            table,
            headers=[
                'bin',
                'N',
                'FFX true gain',
                'FFX gap',
                'FFX dist gain',
                'FFX med dist',
                'FFX cos',
                'sigma gain',
                'sigma gap',
                'sigma dist gain',
                'sigma med dist',
                'blup gap',
                'accept',
                'cap',
            ],
            tablefmt='github',
        )
    )


def _printWorst(rows: list[dict], metric: str, top_k: int, *, reverse: bool = True) -> None:
    selected = sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)[:top_k]
    table = []
    for row in selected:
        table.append(
            [
                row['data_id'],
                row['partition'],
                row['dataset_idx'],
                f'{row[metric]:+.4f}',
                f'{row["raw_ffx_rmse"]:.4f}',
                f'{row["current_ffx_rmse"]:.4f}',
                f'{row["inla_ffx_rmse"]:.4f}',
                f'{row["ffx_distance_gain"]:+.4f}',
                f'{row["sigma_inla_gap"]:+.4f}',
                row['d'],
                row['q'],
                row['m'],
                f'{row["y_mean"]:.3f}',
                f'{row["y_zero_frac"]:.3f}',
                f'{row["sigma_true_mean"]:.3f}',
                f'{row["sigma_current_mean"]:.3f}',
                f'{row["sigma_inla_mean"]:.3f}',
                f'{row["accept"]:.0f}',
                f'{row["sigma_capped"]:.0f}',
                f'{row["sigma_grid_gate"]:.0f}',
                f'{row["sigma_grid_accept"]:.0f}',
                f'{row["marginal_beta_gate"]:.0f}',
            ]
        )
    direction = 'Top' if reverse else 'Bottom'
    print(f'\n{direction} {len(selected)} by {metric}')
    print(
        tabulate(
            table,
            headers=[
                'data',
                'part',
                'idx',
                'metric',
                'raw FFX',
                'EB FFX',
                'INLA FFX',
                'FFX dist gain',
                'sigma gap',
                'd',
                'q',
                'm',
                'y mean',
                'zero',
                'sig true',
                'sig EB',
                'sig INLA',
                'acc',
                'cap',
                'sg_gate',
                'sg_acc',
                'mb_gate',
            ],
            tablefmt='github',
        )
    )


def _writeCsv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwrote row diagnostics to {path}')


def runDiagnostic(args: argparse.Namespace) -> None:
    rows = []
    t0 = time.perf_counter()
    for data_id in args.data_ids:
        cfg = loadDataConfig(data_id)
        if cfg.get('likelihood_family') != 2:
            raise ValueError(f'{data_id} is not a Poisson dataset')
        max_d = cfg['max_d']
        max_q = cfg['max_q']
        paths = [p for p in _pathsForArgs(cfg, args.partition, args.n_epochs) if p.exists()]
        _checkInlaFits(paths, data_id)
        seen = 0
        ok = 0
        for path in paths:
            if seen >= args.n_inla:
                break
            inla_data = _loadInlaFits(path)
            file_n_seen = 0
            for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                if seen >= args.n_inla:
                    break
                if seen + batch['X'].shape[0] > args.n_inla:
                    batch = _sliceBatch(batch, args.n_inla - seen)
                batch = toDevice(batch, torch.device('cpu'))
                stats = _fitPoissonMethods(batch, max_q)
                mask_d, mask_q = _activeMasks(batch, max_d, max_q)
                B = batch['X'].shape[0]
                for b in range(B):
                    if seen >= args.n_inla:
                        break
                    active_d = np.flatnonzero(mask_d[b])
                    active_q = np.flatnonzero(mask_q[b])
                    inla_idx = file_n_seen
                    file_n_seen += 1
                    seen += 1
                    if inla_data['failed'][inla_idx]:
                        continue
                    m_b = int(batch['m'][b].item())
                    est = {
                        'beta': inla_data['beta'][inla_idx, active_d],
                        'sigma_rfx': inla_data['sigma_rfx'][inla_idx, active_q],
                        'blups': inla_data['blups'][inla_idx, :m_b, :][:, active_q],
                    }
                    ok += 1
                    rows.append(
                        _extractRow(
                            data_id,
                            args.partition,
                            seen - 1,
                            batch,
                            stats,
                            b,
                            active_d,
                            active_q,
                            est,
                        )
                    )
        print(f'{data_id}/{args.partition}: INLA ok {ok}/{seen}')

    print(f'\ncompleted {len(rows)} matched rows in {time.perf_counter() - t0:.1f}s')
    if not rows:
        print('No matched rows available. Check that .inla.npz files are present.')
        return
    _printOverall(rows)
    for by in ['data_id', 'd_bin', 'q_bin', 'zero_bin', 'mean_y_bin', 'marginal_beta_gate', 'sigma_grid_gate']:
        _printGrouped(rows, by)
    _printWorst(rows, 'ffx_inla_gap', args.top_k)
    _printWorst(rows, 'sigma_inla_gap', args.top_k)
    _printWorst(rows, 'ffx_distance_gain', args.top_k)
    _printWorst(rows, 'ffx_distance_gain', args.top_k, reverse=False)
    if args.output_csv:
        _writeCsv(rows, Path(args.output_csv))


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-ids', nargs='+', default=DATA_IDS)
    parser.add_argument('--partition', default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs', type=int, default=2)
    parser.add_argument('--n-inla', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output-csv', default='')
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    runDiagnostic(setup())
