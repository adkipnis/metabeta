from pathlib import Path
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.utils.evaluation import EvaluationSummary, getNames, Proposal, dictMean
from metabeta.utils.plot import PALETTE, savePlot, niceify


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

    # final touches
    limits = (min(nominal), max(nominal))
    ax.plot(limits, limits, '--', lw=2, zorder=1, color='grey', alpha=0.5)
    ax.grid(True)
    ax.set_xticks(nominal)
    ax.set_yticks(nominal)

    # niceify
    info = {
        'title': 'Credible Intervals',
        'ylabel': 'Observed',
        'xlabel': 'Nominal',
        'show_title': upper,
        'show_legend': upper,
        'show_x': lower,
        'stats': stats,
        'stats_suffix': '%',
        }
    niceify(ax, info)


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
        savePlot(plot_dir, 'coverage', epoch=epoch)
    plt.show()
    plt.close(fig)
