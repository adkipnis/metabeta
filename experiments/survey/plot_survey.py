"""Histograms of the GLMM literature survey (paper appendix figure).

Reads model_equations.csv next to this script and writes a vector PDF:
    python experiments/survey/plot_survey.py [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from metabeta.utils.plot import DPI, PALETTE

HERE = Path(__file__).resolve().parent
CSV = HERE / 'model_equations.csv'
DEFAULT_OUT = HERE / 'survey-histograms.pdf'

# supported envelope (matches the coverage claim in the README / paper)
MAX_D = 16
MAX_Q = 5
FAMILIES = ['Gaussian', 'Bernoulli', 'Poisson']
LUMP = 17  # counts >= LUMP share one "17+" bar

SUPPORTED = PALETTE[3]  # red
UNSUPPORTED = PALETTE[0]  # blue
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_NOTE = 18, 16, 13, 14, 12


def load() -> list[dict]:
    with CSV.open() as f:
        return list(csv.DictReader(f))


def countBars(values: list[int], lo: int) -> tuple[list[str], list[int], list[bool]]:
    """Per-value counts from lo..LUMP-1 plus one lumped bar; returns labels, counts, raw values."""
    c = Counter(values)
    labels = [str(v) for v in range(lo, LUMP)] + [f'{LUMP}+']
    counts = [c[v] for v in range(lo, LUMP)] + [sum(n for v, n in c.items() if v >= LUMP)]
    raw = list(range(lo, LUMP)) + [LUMP]
    return labels, counts, raw


def barPanel(ax, labels, counts, colors, title, xlabel, note=None):
    x = np.arange(len(labels))
    ax.bar(x, counts, color=colors, width=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    # thin out dense integer axes: label every other tick, always keep the lumped bar
    if len(labels) > 8:
        for i, lab in enumerate(ax.get_xticklabels()):
            if (i % 2 == 1 or i == len(labels) - 2) and i != len(labels) - 1:
                lab.set_visible(False)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE, pad=10)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if note:
        ax.text(
            0.97, 0.95, note, transform=ax.transAxes, ha='right', va='top', fontsize=FS_NOTE,
            color='0.3',
        )


def main(out: Path) -> Path:
    rows = load()
    d = [int(float(r['n_fixed_effects'])) for r in rows if r['n_fixed_effects'] != '']
    q = [int(float(r['n_random_effects'])) for r in rows if r['n_random_effects'] != '']
    fam = [r['family'] if r['family'] in FAMILIES else 'Other' for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)

    labels, counts, raw = countBars(d, 0)
    colors = [SUPPORTED if v <= MAX_D else UNSUPPORTED for v in raw]
    barPanel(
        axes[0], labels, counts, colors, 'Fixed effects', 'number of fixed effects',
        note=f'median {int(np.median(d))}, mean {np.mean(d):.1f}',
    )
    axes[0].set_ylabel('Papers', fontsize=FS_LABEL)

    labels, counts, raw = countBars(q, 1)
    colors = [SUPPORTED if v <= MAX_Q else UNSUPPORTED for v in raw]
    barPanel(
        axes[1], labels, counts, colors, 'Random effects', 'number of random effects',
        note=f'median {int(np.median(q))}, mean {np.mean(q):.1f}',
    )

    fam_labels = FAMILIES + ['Other']
    fam_counts = [fam.count(f) for f in fam_labels]
    colors = [SUPPORTED] * len(FAMILIES) + [UNSUPPORTED]
    barPanel(axes[2], fam_labels, fam_counts, colors, 'Likelihood family', '')
    for i, n in enumerate(fam_counts):
        pct = 100 * n / len(fam)
        axes[2].text(i, n, f'{pct:.1f}%' if pct < 1 else f'{pct:.0f}%', ha='center', va='bottom', fontsize=FS_NOTE)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SUPPORTED),
        plt.Rectangle((0, 0), 1, 1, color=UNSUPPORTED),
    ]
    axes[2].legend(
        handles, ['supported by metabeta', 'not supported'], fontsize=FS_LEGEND, frameon=False,
        loc='upper right',
    )

    fig.tight_layout(w_pad=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    print(main(parser.parse_args().out))
