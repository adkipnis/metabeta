"""
Warm-start efficiency figure.

Panel B: ESS/s vs n_params — NUTS (cold_std) vs MB-NUTS (warm_2000)

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

from matplotlib import pyplot as plt

from metabeta.utils.plot import DPI, savePlot
from metabeta.utils.warmfit import COND_STYLE, collectWarmRecords, plotWarmPanel


def plotWarmStart(
    dirs: list[Path],
    fits_tag: str,
    conds: list[str] | None = None,
    out_dir: Path | None = None,
    n_bins: int = 8,
    show: bool = False,
) -> Path | None:
    if conds is None:
        conds = ['cold_std', 'warm_2000']

    records: list[dict] = []
    for d in dirs:
        if not (d / fits_tag).exists():
            print(f'[warn] fits dir not found: {d / fits_tag} — skipping')
            continue
        records.extend(collectWarmRecords(d, fits_tag, conds))

    if not records:
        raise ValueError('No records collected — check dirs and fits_tag.')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=DPI)
    plotWarmPanel(
        ax,
        records,
        'ess_s',
        conds,
        COND_STYLE,
        'Effective samples / second',
        'Sampling efficiency',
        n_bins,
        log_y=False,
    )
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
    p.add_argument('--conds',    nargs='+', default=['cold_std', 'warm_2000'])
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
