"""
Warm-start efficiency figure.

Panel B: ESS/s vs n_params — NUTS (cold_std) vs MB-NUTS (warm_1000)

X-axis: n_params = d + q + m*q (effective model parameters per dataset)
Line + band: equal-count bins in n_params space → median + 5th/95th percentile

Usage (from repo root):
    uv run python -m metabeta.plotting.warmstart \\
        --dirs metabeta/outputs/data/tiny-n-sampled \\
        --fits_tag fits_warm_normal_dsmall-n-mixed_mlarge-r_s3 \\
        --out_dir experiments/results/warm_start
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.utils.dataloader import Collection
from metabeta.utils.plot import DPI, PALETTE, savePlot, niceify

COND_STYLE: dict[str, dict] = {
    'cold_std':  {'color': PALETTE[3], 'label': 'NUTS'},
    'cold_gold': {'color': PALETTE[5], 'label': 'NUTS (gold)'},
    'warm_2000': {'color': PALETTE[4], 'label': 'MB-NUTS (2000)'},
    'warm_1000': {'color': PALETTE[4], 'label': 'MB-NUTS'},
    'warm_500':  {'color': PALETTE[4], 'label': 'MB-NUTS (500)'},
    'warm_250':  {'color': PALETTE[4], 'label': 'MB-NUTS (250)'},
}


def _nParams(d: int, q: int, m: int) -> int:
    return d + q + m * q


def _loadFits(fits_dir: Path, cond: str, n: int) -> list[dict | None]:
    fits: list[dict | None] = []
    for idx in range(n):
        path = fits_dir / f'{cond}__{idx:03d}.npz'
        if not path.exists():
            fits.append(None)
            continue
        with np.load(path) as f:
            fits.append(
                {
                    'wall_s': float(f['wall_s']),
                    'min_ess': float(f['min_ess']),
                    'n_div': float(f['n_div']),
                    'max_rhat': float(f['max_rhat']),
                }
            )
    return fits


def _collectRecords(data_dir: Path, fits_tag: str, conds: list[str]) -> list[dict]:
    col = Collection(data_dir / 'valid.npz', permute=False)
    n = len(col)
    raw = col.raw
    fits_dir = data_dir / fits_tag

    fits_by_cond = {c: _loadFits(fits_dir, c, n) for c in conds}
    records: list[dict] = []
    for idx in range(n):
        n_params = _nParams(int(raw['d'][idx]), int(raw['q'][idx]), int(raw['m'][idx]))
        for cond in conds:
            fit = fits_by_cond[cond][idx]
            if fit is None:
                continue
            records.append(
                {
                    'data_dir': data_dir.name,
                    'idx': idx,
                    'cond': cond,
                    'n_params': n_params,
                    'wall_s': fit['wall_s'],
                    'ess_s': fit['min_ess'] / max(fit['wall_s'], 1e-3),
                    'n_div': fit['n_div'],
                    'max_rhat': fit['max_rhat'],
                }
            )
    return records


def _binStats(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-count bins. Returns (centers, medians, lo, hi); bins with < 2 pts are dropped."""
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


def _plotPanel(
    ax: Axes,
    records: list[dict],
    metric: str,
    conds: list[str],
    ylabel: str,
    title: str,
    n_bins: int,
    log_y: bool = False,
) -> None:
    for cond in conds:
        style = COND_STYLE.get(cond, {'color': 'grey', 'label': cond})
        sub = [r for r in records if r['cond'] == cond]
        if not sub:
            continue
        x = np.array([r['n_params'] for r in sub], dtype=float)
        y = np.array([r[metric] for r in sub], dtype=float)

        ax.scatter(x, y, color=style['color'], alpha=0.2, s=25, zorder=2, linewidths=0)

        centers, meds, los, his = _binStats(x, y, n_bins)
        if len(centers) == 0:
            continue
        ax.plot(
            centers, meds, '-o',
            color=style['color'], lw=2.0, ms=6, zorder=3, label=style['label'],
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
            'legend_loc': 'upper left',
            'despine': True,
            'grid_alpha': 1,
        },
    )


def plotWarmStart(
    dirs: list[Path],
    fits_tag: str,
    conds: list[str] | None = None,
    out_dir: Path | None = None,
    n_bins: int = 8,
    show: bool = False,
) -> Path | None:
    if conds is None:
        conds = ['cold_std', 'warm_1000']

    records: list[dict] = []
    for d in dirs:
        if not (d / fits_tag).exists():
            print(f'[warn] fits dir not found: {d / fits_tag} — skipping')
            continue
        records.extend(_collectRecords(d, fits_tag, conds))

    if not records:
        raise ValueError('No records collected — check dirs and fits_tag.')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=DPI)
    _plotPanel(ax, records, 'ess_s', conds, 'Effective samples / second', 'Sampling efficiency', n_bins)
    fig.tight_layout()

    saved = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = savePlot(out_dir, 'warmstart')
        savePlot(out_dir, 'warmstart', ending='pdf')
    if show:
        plt.show()
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# fmt: off
def _setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot warm-start diagnostics.')
    p.add_argument('--dirs',     nargs='+', type=Path, required=True,
                   help='data directories (e.g. metabeta/outputs/data/tiny-n-sampled)')
    p.add_argument('--fits_tag', required=True,
                   help='fits subdirectory, e.g. fits_warm_normal_dsmall-n-mixed_mlarge-r_s3')
    p.add_argument('--conds',    nargs='+', default=['cold_std', 'warm_1000'])
    p.add_argument('--out_dir',  type=Path, default=None)
    p.add_argument('--n_bins',   type=int, default=8)
    p.add_argument('--show',     action='store_true')
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    args = _setup()
    path = plotWarmStart(
        dirs=args.dirs,
        fits_tag=args.fits_tag,
        conds=args.conds,
        out_dir=args.out_dir,
        n_bins=args.n_bins,
        show=args.show,
    )
    if path:
        print(f'Saved → {path}')
