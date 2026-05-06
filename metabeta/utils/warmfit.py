"""
Shared utilities for warm-start fit caching and plotting.

Cache format: {fits_dir}/{cond_label}__{idx:03d}.npz
  - sample arrays: ffx, sigma_rfx, rfx, sigma_eps (optional)
  - diagnostics: wall_s, n_div, max_rhat, min_ess, min_ess_t
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from metabeta.utils.dataloader import Collection
from metabeta.utils.plot import PALETTE, niceify

COND_STYLE: dict[str, dict] = {
    'mb': {'color': PALETTE[0], 'label': 'MB'},
    'cold_std': {'color': PALETTE[3], 'label': 'NUTS'},
    'warm_2000': {'color': PALETTE[4], 'label': 'MB-NUTS (2000)'},
    'warm_1000': {'color': PALETTE[4], 'label': 'MB-NUTS'},
    'warm_500': {'color': PALETTE[4], 'label': 'MB-NUTS (500)'},
    'warm_250': {'color': PALETTE[4], 'label': 'MB-NUTS (250)'},
    'advi': {'color': PALETTE[1], 'label': 'ADVI'},
}


def nParams(d: int, q: int, m: int) -> int:
    return d + q + m * q


def cachePath(fits_dir: Path, cond_label: str, idx: int) -> Path:
    return fits_dir / f'{cond_label}__{idx:03d}.npz'


def saveFit(path: Path, samples: dict[str, np.ndarray], diag: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **samples, **{k: np.array(v) for k, v in diag.items()})


def loadFit(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return (samples_dict, diag_dict). Unknown keys are silently ignored."""
    with np.load(path) as f:
        raw = dict(f)
    sample_keys = {'ffx', 'sigma_rfx', 'sigma_eps', 'rfx'}
    diag_keys = {'n_div', 'max_rhat', 'min_ess', 'min_ess_t', 'wall_s'}
    return (
        {k: raw[k] for k in sample_keys if k in raw},
        {k: float(raw[k]) for k in diag_keys if k in raw},
    )


def binStats(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-count bins → (centers, medians, lo, hi); bins with < 2 pts are dropped."""
    edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    centers, meds, los, his = [], [], [], []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        mask = (x >= edges[i]) & (x <= edges[i + 1] if last else x < edges[i + 1])
        vals = y[mask]
        if len(vals) < 2:
            continue
        centers.append(float(np.median(x[mask])))
        meds.append(float(np.median(vals)))
        los.append(float(np.percentile(vals, lo_pct)))
        his.append(float(np.percentile(vals, hi_pct)))
    return np.array(centers), np.array(meds), np.array(los), np.array(his)


def collectWarmRecords(data_dir: Path, fits_tag: str, conds: list[str]) -> list[dict]:
    """Load per-dataset records from test.npz + fits cache.

    Each record has: data_dir, idx, cond, n_params, wall_s, and any available
    diagnostics (ess_s, n_div, max_rhat, min_ess, min_ess_t). Missing fits are
    skipped silently.
    """
    col = Collection(data_dir / 'test.npz', permute=False)
    n = len(col)
    raw = col.raw
    fits_dir = data_dir / fits_tag

    records: list[dict] = []
    for idx in range(n):
        n_p = nParams(int(raw['d'][idx]), int(raw['q'][idx]), int(raw['m'][idx]))
        for cond in conds:
            path = cachePath(fits_dir, cond, idx)
            if not path.exists():
                continue
            _, diag = loadFit(path)
            records.append(
                {
                    'data_dir': data_dir.name,
                    'idx': idx,
                    'cond': cond,
                    'n_params': n_p,
                    **diag,
                    'ess_s': diag['min_ess'] / max(diag.get('wall_s', 1e-3), 1e-3)
                    if 'min_ess' in diag
                    else float('nan'),
                }
            )
    return records


def plotWarmPanel(
    ax: Axes,
    records: list[dict],
    metric: str,
    conds: list[str],
    cond_style: dict[str, dict],
    ylabel: str,
    title: str,
    n_bins: int,
    log_y: bool = False,
    legend_loc: str = 'upper left',
) -> None:
    for cond in conds:
        style = cond_style.get(cond, {'color': 'grey', 'label': cond})
        sub = [r for r in records if r['cond'] == cond]
        if not sub:
            continue
        x = np.array([r['n_params'] for r in sub], dtype=float)
        y = np.array([r[metric] for r in sub], dtype=float)

        ax.scatter(x, y, color=style['color'], alpha=0.2, s=25, zorder=2, linewidths=0)

        # Fewer bins for sparse conditions to avoid gaps from tied x-values collapsing bins
        cond_bins = max(2, min(n_bins, len(sub) // 4))
        centers, meds, los, his = binStats(x, y, cond_bins)
        if len(centers) == 0:
            continue
        ax.plot(
            centers,
            meds,
            '-o',
            color=style['color'],
            lw=2.0,
            ms=6,
            zorder=3,
            label=style['label'],
        )
        ax.fill_between(centers, los, his, color=style['color'], alpha=0.15, zorder=1)

    if log_y:
        ax.set_yscale('log')
    niceify(
        ax,
        {
            'title': title,
            'xlabel': '# parameters',
            'ylabel': ylabel,
            'title_fs': 24,
            'xlabel_fs': 20,
            'ylabel_fs': 20,
            'ticks_ls': 16,
            'show_legend': True,
            'legend_fs': 16,
            'legend_ms': 1.5,
            'legend_loc': legend_loc,
            'despine': True,
            'grid_alpha': 1,
        },
    )
