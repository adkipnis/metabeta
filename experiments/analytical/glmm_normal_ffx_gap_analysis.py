"""Per-dataset FFX gap analysis: current vs precomputed INLA.

Distinguishes outlier-driven from systematic FFX gaps. Uses precomputed
.inla.npz files — no live INLA calls needed.

Key outputs:
  - per-dataset gap distribution (percentiles, fraction worse)
  - concentration: how much of the aggregate gap sits in the worst N%
  - gap breakdown by d_bin, tail_gate, prior_cap, data_id
  - top-k worst datasets

Usage:
    uv run python -u experiments/analytical/glmm_normal_ffx_gap_analysis.py \
        --data-ids large-n-mixed --partition train --max-datasets 1000

    uv run python -u experiments/analytical/glmm_normal_ffx_gap_analysis.py \
        --data-ids small-n-mixed medium-n-mixed large-n-mixed huge-n-mixed \
        --partition train --max-datasets 1000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@dataclass
class Row:
    data_id: str
    idx: int
    d: int
    q: int
    m: int
    n: int
    current_rmse: float  # sqrt(mean(β_err²)) per dataset
    inla_rmse: float
    gap: float  # current_rmse - inla_rmse
    tail_gate: bool  # normal_beta_tail_grid_gate fired
    prior_capped: bool  # normal_map_beta_prior_capped fired
    stabilized: bool  # normal_map_beta_stabilized fired
    eb_accept: float  # normal_laplace_eb_accept rate
    sigma_rfx_gap: float
    blup_gap: float


def _paths(cfg: dict, partition: str, n_epochs: int) -> list[Path]:
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _inla_path(data_path: Path) -> Path:
    return data_path.with_suffix('.inla.npz')


def _rmse(errs: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errs**2))) if errs.size else float('nan')


def _collect(
    data_id: str,
    partition: str,
    n_epochs: int,
    max_datasets: int | None,
    batch_size: int,
) -> list[Row]:
    cfg = loadDataConfig(data_id)
    max_q = cfg['max_q']
    paths = _paths(cfg, partition, n_epochs)

    rows: list[Row] = []
    n_seen = 0

    with torch.no_grad():
        for path in paths:
            if max_datasets is not None and n_seen >= max_datasets:
                break
            inla_path = _inla_path(path)
            if not inla_path.exists():
                print(f'  WARN: missing {inla_path}, skipping')
                continue
            with np.load(inla_path) as f:
                inla_ffx = f['inla_ffx'].copy()  # (N, max_d)
                inla_srfx = f['inla_sigma_rfx'].copy()  # (N, max_q)
                inla_rfx = f['inla_rfx'].copy()  # (N, max_m, max_q)
                inla_failed = f['inla_failed'].copy().astype(bool)  # (N,)

            file_idx = 0
            for batch in Dataloader(path, batch_size=batch_size, shuffle=False, sortish=False):
                if max_datasets is not None and n_seen >= max_datasets:
                    break
                if max_datasets is not None:
                    remaining = max_datasets - n_seen
                    if batch['X'].shape[0] > remaining:
                        batch = _sliceBatch(batch, remaining)
                batch = toDevice(batch, torch.device('cpu'))
                B = batch['X'].shape[0]
                Zm = batch['Z'][..., :max_q]

                stats = glmm(
                    batch['X'],
                    batch['y'],
                    Zm,
                    batch['mask_n'].float(),
                    batch['mask_m'].float(),
                    batch['ns'].clamp(min=1).float(),
                    batch['n'].float(),
                    likelihood_family=0,
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

                ffx_true = batch['ffx'].cpu().numpy()
                srfx_true = batch['sigma_rfx'].cpu().numpy()
                rfx_true = batch['rfx'].cpu().numpy()
                mask_d = batch['mask_d'].cpu().numpy().astype(bool)
                mask_q = batch['mask_q'].cpu().numpy().astype(bool)
                m_arr = batch['m'].cpu().numpy().astype(int)
                n_arr = batch['n'].cpu().numpy().astype(int)

                beta_est = stats['beta_est'].cpu().numpy()
                srfx_est = stats['sigma_rfx_est'].cpu().numpy()
                blup_est = stats['blup_est'].cpu().numpy()
                _zeros = torch.zeros(B)
                tail_gate = stats.get('normal_beta_tail_grid_gate', _zeros).cpu().numpy()
                prior_capped = stats.get('normal_map_beta_prior_capped', _zeros).cpu().numpy()
                stabilized = stats.get('normal_map_beta_stabilized', _zeros).cpu().numpy()
                eb_accept = stats.get('normal_laplace_eb_accept', _zeros).cpu().numpy()

                for b in range(B):
                    global_idx = file_idx + b
                    active_d = np.flatnonzero(mask_d[b])
                    active_q = np.flatnonzero(mask_q[b])
                    m = m_arr[b]
                    n = n_arr[b]

                    curr_beta_err = beta_est[b, active_d] - ffx_true[b, active_d]
                    curr_srfx_err = srfx_est[b, active_q] - srfx_true[b, active_q]
                    rfx_act = rfx_true[b, :m][:, active_q]
                    curr_blup_err = (blup_est[b, :m][:, active_q] - rfx_act).reshape(-1)

                    if inla_failed[global_idx]:
                        inla_beta_rmse = float('nan')
                        inla_srfx_rmse = float('nan')
                        inla_blup_rmse = float('nan')
                    else:
                        inla_beta_err = inla_ffx[global_idx, active_d] - ffx_true[b, active_d]
                        inla_beta_rmse = _rmse(inla_beta_err)
                        inla_srfx_err = inla_srfx[global_idx, active_q] - srfx_true[b, active_q]
                        inla_srfx_rmse = _rmse(inla_srfx_err)
                        inla_rfx_act = inla_rfx[global_idx, :m][:, active_q]
                        inla_blup_rmse = _rmse((inla_rfx_act - rfx_act).reshape(-1))

                    curr_beta_rmse = _rmse(curr_beta_err)
                    curr_srfx_rmse = _rmse(curr_srfx_err)
                    curr_blup_rmse = _rmse(curr_blup_err)

                    rows.append(
                        Row(
                            data_id=data_id,
                            idx=n_seen + b,
                            d=int(active_d.size),
                            q=int(active_q.size),
                            m=m,
                            n=n,
                            current_rmse=curr_beta_rmse,
                            inla_rmse=inla_beta_rmse,
                            gap=curr_beta_rmse - inla_beta_rmse,
                            tail_gate=bool(tail_gate[b] > 0.5),
                            prior_capped=bool(prior_capped[b] > 0.5),
                            stabilized=bool(stabilized[b] > 0.5),
                            eb_accept=float(eb_accept[b]),
                            sigma_rfx_gap=curr_srfx_rmse - inla_srfx_rmse,
                            blup_gap=curr_blup_rmse - inla_blup_rmse,
                        )
                    )

                n_seen += B
                file_idx += B

    return rows


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    B = batch['X'].shape[0]
    return {
        k: v[:n] if (torch.is_tensor(v) and v.shape[:1] == (B,)) else v for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _binByD(d: int) -> str:
    if d <= 4:
        return 'd<=4'
    if d <= 8:
        return '5-8'
    if d <= 12:
        return '9-12'
    return 'd>=13'


def _pct(xs: np.ndarray, q: float) -> str:
    v = float(np.nanpercentile(xs, q))
    return f'{v:+.3f}'


def _gapDistribution(gaps: np.ndarray, label: str) -> list[list]:
    finite = gaps[np.isfinite(gaps)]
    if finite.size == 0:
        return []
    frac_worse = float(np.mean(finite > 0.0))
    frac_tied = float(np.mean(np.abs(finite) < 0.01))
    frac_mildly_worse = float(np.mean((finite >= 0.01) & (finite < 0.05)))
    frac_significantly_worse = float(np.mean(finite >= 0.05))
    return [
        [
            label,
            int(finite.size),
            f'{float(np.nanmean(finite)):+.4f}',
            f'{float(np.nanmedian(finite)):+.4f}',
            _pct(finite, 75),
            _pct(finite, 90),
            _pct(finite, 95),
            f'{frac_worse:.2f}',
            f'{frac_tied:.2f}',
            f'{frac_mildly_worse:.2f}',
            f'{frac_significantly_worse:.2f}',
        ]
    ]


def _concentration(rows: list[Row], label: str) -> None:
    """Report what fraction of aggregate MSE gap comes from worst P% of datasets."""
    valid = [r for r in rows if np.isfinite(r.gap)]
    if not valid:
        return
    gaps = np.array([r.gap for r in valid])
    curr = np.array([r.current_rmse for r in valid])
    inla = np.array([r.inla_rmse for r in valid])

    # contribution to aggregate MSE gap per dataset (proportional to d)
    d_arr = np.array([r.d for r in valid], dtype=float)
    contrib = (curr**2 - inla**2) * d_arr  # MSE gap * d (weight by num β coeffs)
    total_gap = float(np.sum(contrib))

    sorted_contrib = np.sort(contrib)[::-1]
    cum = np.cumsum(sorted_contrib)

    table = []
    for pct in [5, 10, 20, 50]:
        k = max(1, int(np.ceil(len(valid) * pct / 100)))
        top_k_share = float(cum[k - 1]) / max(abs(total_gap), 1e-10)
        table.append([f'top {pct}%', k, f'{top_k_share:.2f}'])
    print(f'\nConcentration — {label} (share of aggregate MSE gap)')
    print(tabulate(table, headers=['group', 'N', 'share of gap'], tablefmt='simple'))


def _groupedBreakdown(rows: list[Row], key_fn, label: str) -> None:
    valid = [r for r in rows if np.isfinite(r.gap)]
    if not valid:
        return
    keys = sorted({key_fn(r) for r in valid})
    table: list[list] = []
    for k in keys:
        sel = [r for r in valid if key_fn(r) == k]
        gaps = np.array([r.gap for r in sel])
        table += _gapDistribution(gaps, k)
    print(f'\nGap distribution grouped by {label}')
    print(
        tabulate(
            table,
            headers=[
                label,
                'N',
                'mean Δ',
                'med Δ',
                'p75',
                'p90',
                'p95',
                'frac>0',
                'frac≈0',
                'mild',
                'large',
            ],
            tablefmt='github',
        )
    )


def _topWorst(rows: list[Row], k: int = 20) -> None:
    valid = sorted([r for r in rows if np.isfinite(r.gap)], key=lambda r: r.gap, reverse=True)[:k]
    table = [
        [
            r.data_id,
            r.idx,
            f'{r.gap:+.4f}',
            f'{r.current_rmse:.4f}',
            f'{r.inla_rmse:.4f}',
            r.d,
            r.q,
            r.m,
            int(r.tail_gate),
            int(r.prior_capped),
            int(r.stabilized),
            f'{r.eb_accept:.2f}',
        ]
        for r in valid
    ]
    print(f'\nTop {len(valid)} worst datasets by FFX gap')
    print(
        tabulate(
            table,
            headers=[
                'data_id',
                'idx',
                'gap',
                'curr RMSE',
                'INLA RMSE',
                'd',
                'q',
                'm',
                'tail',
                'cap',
                'stab',
                'eb_acc',
            ],
            tablefmt='github',
        )
    )


def _summary(rows: list[Row]) -> None:
    valid = [r for r in rows if np.isfinite(r.gap)]
    if not valid:
        print('No valid rows.')
        return
    gaps = np.array([r.gap for r in valid])
    print('\nOverall FFX gap distribution:')
    print(
        tabulate(
            _gapDistribution(gaps, 'all'),
            headers=[
                'group',
                'N',
                'mean Δ',
                'med Δ',
                'p75',
                'p90',
                'p95',
                'frac>0',
                'frac≈0',
                'mild',
                'large',
            ],
            tablefmt='simple',
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    all_rows: list[Row] = []
    t0 = time.perf_counter()
    for data_id in args.data_ids:
        cfg = loadDataConfig(data_id)
        partition = args.partition
        print(f'Collecting {data_id}/{partition} ...', flush=True)
        rows = _collect(
            data_id,
            partition,
            n_epochs=args.n_epochs,
            max_datasets=args.max_datasets,
            batch_size=args.batch_size,
        )
        n_valid = sum(1 for r in rows if np.isfinite(r.gap))
        n_inla_fail = sum(1 for r in rows if not np.isfinite(r.gap))
        print(f'  {len(rows)} datasets, {n_valid} with INLA, {n_inla_fail} INLA failed')
        all_rows.extend(rows)
    print(f'\nTotal collection: {len(all_rows)} datasets in {time.perf_counter()-t0:.1f}s')

    _summary(all_rows)
    _concentration(all_rows, 'all datasets')
    _groupedBreakdown(all_rows, lambda r: r.data_id, 'data_id')
    _groupedBreakdown(all_rows, lambda r: _binByD(r.d), 'd_bin')
    _groupedBreakdown(
        all_rows,
        lambda r: 'tail_gate' if r.tail_gate else 'no_tail_gate',
        'tail_gate',
    )
    _groupedBreakdown(
        all_rows,
        lambda r: 'cap' if r.prior_capped else 'no_cap',
        'prior_cap',
    )
    _groupedBreakdown(
        all_rows,
        lambda r: 'stab' if r.stabilized else 'no_stab',
        'stabilized',
    )
    _topWorst(all_rows, k=args.top_k)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-ids',     nargs='+', default=['large-n-mixed'])
    parser.add_argument('--partition',               default='train')
    parser.add_argument('--n-epochs',     type=int,  default=2)
    parser.add_argument('--max-datasets', type=int,  default=1000)
    parser.add_argument('--batch-size',   type=int,  default=32)
    parser.add_argument('--top-k',        type=int,  default=20)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run(setup())
