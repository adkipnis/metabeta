"""
Runtime vs n_params figure (Panel A of warm-start figure).

Lines: MB | MB-NUTS (warm_2000) | NUTS | ADVI
X-axis: n_params = d + q + m*q (effective model parameters per dataset)
Line + band: equal-count bins → median + 5th/95th percentile

Data sources:
- NUTS (cold_std) and ADVI wall times: test.fit.npz (nuts_duration / advi_duration),
  all 512 datasets per data_dir
- MB-NUTS wall times: fits_dir/{cond}__{idx:03d}.npz (wall_s), indexed against test.npz
- MB wall times: fits_dir/mb__{idx:03d}.npz (wall_s), n_params inferred from sample shapes

Usage (from repo root):
    uv run python -m metabeta.plotting.runtimes \\
        --dirs metabeta/outputs/data/tiny-n-sampled \\
        --fits_tag fits_warm_normal_dsmall-n-mixed_mlarge-r_s3 \\
        --out_dir experiments/results/warm_start
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from metabeta.utils.plot import DPI, savePlot
from metabeta.utils.warmfit import (
    COND_STYLE,
    collectWarmRecords,
    nParams,
    plotWarmPanel,
)

_WARM_CONDS = frozenset(COND_STYLE) - {'mb', 'advi', 'cold_std'}
_DEFAULT_CONDS = ['mb', 'warm_2000', 'cold_std', 'advi']
_METHOD_TO_COND = {'metabeta': 'mb', 'NUTS': 'cold_std', 'ADVI': 'advi'}


def _collectRuntimeRecords(data_dir: Path, fits_tag: str, conds: list[str]) -> list[dict]:
    # Warm MCMC conditions (warm_X): load from per-dataset cache files
    warm_conds = [c for c in conds if c in _WARM_CONDS]
    records: list[dict] = collectWarmRecords(data_dir, fits_tag, warm_conds) if warm_conds else []

    fits_dir = data_dir / fits_tag

    # cold_std (NUTS) and ADVI: load directly from test.fit.npz (all datasets)
    test_fit = data_dir / 'test.fit.npz'
    if test_fit.exists() and any(c in conds for c in ('cold_std', 'advi')):
        with np.load(test_fit, allow_pickle=True) as raw:
            ds, qs, ms = raw['d'], raw['q'], raw['m']
            nuts_dur = raw['nuts_duration'] if 'cold_std' in conds else None
            advi_dur = raw['advi_duration'] if 'advi' in conds else None
            for i in range(len(ds)):
                n_p = nParams(int(ds[i]), int(qs[i]), int(ms[i]))
                if nuts_dur is not None:
                    records.append(
                        {
                            'data_dir': data_dir.name,
                            'idx': i,
                            'cond': 'cold_std',
                            'n_params': n_p,
                            'wall_s': float(nuts_dur[i]),
                        }
                    )
                if advi_dur is not None:
                    records.append(
                        {
                            'data_dir': data_dir.name,
                            'idx': i,
                            'cond': 'advi',
                            'n_params': n_p,
                            'wall_s': float(advi_dur[i]),
                        }
                    )

    if 'mb' in conds:
        for p in sorted(fits_dir.glob('mb__*.npz')):
            with np.load(p) as f:
                d = int(f['ffx'].shape[1])
                q = int(f['sigma_rfx'].shape[1])
                m = int(f['rfx'].shape[1])
                wall_s = float(f['wall_s'])
            records.append(
                {
                    'data_dir': data_dir.name,
                    'cond': 'mb',
                    'n_params': nParams(d, q, m),
                    'wall_s': wall_s,
                }
            )

    return records


def plotRuntimes(
    dirs: list[Path],
    fits_tag: str,
    conds: list[str] | None = None,
    out_dir: Path | None = None,
    n_bins: int = 8,
    log_y: bool = True,
    show: bool = False,
    title: str = 'runtimes',
) -> Path | None:
    if conds is None:
        conds = _DEFAULT_CONDS

    records: list[dict] = []
    for d in dirs:
        has_fits = (d / fits_tag).exists()
        has_test_fit = (d / 'test.fit.npz').exists()
        if not has_fits and not has_test_fit:
            print(f'[warn] no data found for {d} — skipping')
            continue
        records.extend(_collectRuntimeRecords(d, fits_tag, conds))

    if not records:
        raise ValueError('No records collected — check dirs and fits_tag.')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=DPI)
    plotWarmPanel(
        ax,
        records,
        'wall_s',
        conds,
        COND_STYLE,
        'Wall time (s)',
        'Runtime',
        n_bins,
        log_y=log_y,
        legend_loc='center',
        center='mean',
        lo_pct=0.5,
        hi_pct=99.5,
        smooth_band=True,
        line_lw=4.0,
        band_alpha=0.22,
        scatter_alpha=0.0,
        x_range=(0, 300),
        plain_log_y_ticks=True,
    )
    fig.tight_layout()

    saved = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = savePlot(out_dir, title)
        savePlot(out_dir, title, ending='pdf')
    if show:
        plt.show()
    plt.close(fig)
    return saved


def plotRuntimeRecords(
    records: list[dict],
    out_dir: Path | None = None,
    n_bins: int = 8,
    log_y: bool = True,
    show: bool = False,
    title: str = 'runtimes',
    config: str | None = 'large-n-mixed',
    x_key: str = 'n_params',
) -> Path | None:
    """Plot runtime experiment records produced by experiments/runtimes.py."""
    if x_key not in {'n_params', 'n'}:
        raise ValueError(f'unknown x_key: {x_key}')

    if config is not None:
        records = [r for r in records if r.get('config') == config]

    plot_records = []
    for r in records:
        cond = _METHOD_TO_COND.get(r['method'])
        if cond is None:
            continue
        plot_records.append(
            {
                'data_dir': r['source'],
                'idx': r.get('idx'),
                'cond': cond,
                'n_params': r['n_params'],
                'n': r['n'],
                'wall_s': r['duration'],
            }
        )

    if not plot_records:
        raise ValueError('No plottable runtime records collected.')

    conds = ['mb', 'cold_std', 'advi']
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=DPI)
    xlabel = '# observations' if x_key == 'n' else '# parameters'
    x_range = (0, 300) if x_key == 'n_params' else (0, max(r[x_key] for r in plot_records))
    plotWarmPanel(
        ax,
        plot_records,
        'wall_s',
        conds,
        COND_STYLE,
        'Wall time (s)',
        'Runtime',
        n_bins,
        x_metric=x_key,
        xlabel=xlabel,
        log_y=log_y,
        legend_loc='center',
        center='mean',
        lo_pct=0.5,
        hi_pct=99.5,
        smooth_band=True,
        line_lw=4.0,
        band_alpha=0.22,
        scatter_alpha=0.0,
        x_range=x_range,
        plain_log_y_ticks=True,
    )
    fig.tight_layout()

    saved = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = savePlot(out_dir, title)
        savePlot(out_dir, title, ending='pdf')
    if show:
        plt.show()
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# fmt: off
def _setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot runtime vs model complexity.')
    p.add_argument('--dirs',     nargs='+', type=Path, required=True,
                   help='data directories (e.g. metabeta/outputs/data/tiny-n-sampled)')
    p.add_argument('--fits_tag', required=True,
                   help='fits subdirectory, e.g. fits_warm_normal_dsmall-n-mixed_mlarge-r_s3')
    p.add_argument('--conds',    nargs='+', default=_DEFAULT_CONDS)
    p.add_argument('--out_dir',  type=Path, default=None)
    p.add_argument('--n_bins',   type=int, default=8)
    p.add_argument('--no_log_y', action='store_true', help='linear y-axis instead of log')
    p.add_argument('--show',     action='store_true')
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    args = _setup()
    path = plotRuntimes(
        dirs=args.dirs,
        fits_tag=args.fits_tag,
        conds=args.conds,
        out_dir=args.out_dir,
        n_bins=args.n_bins,
        log_y=not args.no_log_y,
        show=args.show,
    )
    if path:
        print(f'Saved → {path}')
