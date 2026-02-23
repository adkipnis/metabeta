from pathlib import Path
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.utils.evaluation import EvaluationSummary, getNames, Proposal, dictMean
from metabeta.utils.plot import PALETTE


def _plotCoverage(
    ax: Axes,
    cvrg: dict[float, dict[str, torch.Tensor]],
    names: list[str],
    stats: dict[str, float],
    upper: bool = True,
    lower: bool = True,
) -> None:
    # prepare data
    cols = [torch.cat(list(v.values())).unsqueeze(1) for v in cvrg.values()]
    matrix = torch.cat(cols, dim=1)
    assert len(names) == len(matrix), 'shape mismatch for names'
    nominal = [int(100.0 * (1.0 - alpha)) for alpha in cvrg]

    # plot coverage per parameter
    for i, values in enumerate(matrix):
        coverage_i = 100.0 * values
        ax.plot(nominal, coverage_i, label=names[i], color=PALETTE[i], alpha=0.8, lw=3)

    # stats info
    txt = [f'{k} = {v:.2f}%' for k, v in stats.items()]
    ax.text(
        0.70,
        0.05,
        '\n'.join(txt),
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=20,
        bbox=dict(
            boxstyle='round',
            facecolor=(1, 1, 1, 0.7),
            edgecolor=(0, 0, 0, 0.15),
        ),
    )

    # final touches
    limits = (min(nominal), max(nominal))
    ax.plot(limits, limits, '--', lw=2, zorder=1, color='grey')
    ax.grid(True)
    ax.set_xticks(nominal)
    ax.set_yticks(nominal)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_ylabel('Observed', fontsize=26, labelpad=10)
    if upper:
        ax.set_title('Coverage', fontsize=28, pad=15)
        ax.legend(fontsize=18, loc='upper left')
        if not lower:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelcolor='w', size=1)
    if lower:
        ax.set_xlabel('Nominal', fontsize=26, labelpad=10)


def plotCoverage(
    summary: EvaluationSummary,
    proposal: Proposal,
    plot_dir: Path | None = None,
    epoch: int | None = None,
) -> None:
    names = (
        getNames('ffx', proposal.d) + getNames('sigmas', proposal.q) + getNames('rfx', proposal.q)
    )
    stats = {
        'ECE': 100 * dictMean(summary.ece),
        'LCR': 100 * dictMean(summary.lcr),
    }
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    _plotCoverage(ax, summary.coverage, names, stats)
    ax.set_box_aspect(1)

    # store
    if plot_dir is not None:
        fname = plot_dir / 'coverage_latest.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
        if epoch is not None:
            fname_e = plot_dir / f'coverage_e{epoch}.png'
            plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15)
    plt.show()
    plt.close(fig)
