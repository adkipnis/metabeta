from pathlib import Path
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.utils.evaluation import (
    EvaluationSummary,
    getNames,
    getCorrRfxNames,
    Proposal,
    dictMeanExcl,
)
from metabeta.utils.plot import DPI, PALETTE, savePlot, niceify


def _plotCoverage(
    ax: Axes,
    cvrg: dict[float, dict[str, torch.Tensor]],
    names: list[str],
    stats: dict[str, float],
    title: str | None = 'Credible Intervals',
    show_legend: bool = True,
    show_x: bool = True,
    show_corr_rfx: bool = False,
) -> None:
    # prepare data
    if not show_corr_rfx:
        cvrg = {alpha: {k: v for k, v in d.items() if k != 'corr_rfx'} for alpha, d in cvrg.items()}
    cols = [torch.cat(list(v.values())).unsqueeze(1) for v in cvrg.values()]
    matrix = torch.cat(cols, dim=1)
    assert len(names) == len(matrix), 'shape mismatch for names'
    nominal = [int(100.0 * (1.0 - alpha)) for alpha in cvrg]

    # plot coverage per parameter
    n_colors = len(PALETTE)
    for i, values in enumerate(matrix):
        coverage_i = 100.0 * values
        ax.plot(nominal, coverage_i, label=names[i], color=PALETTE[i % n_colors], alpha=0.8, lw=3)

    # final touches
    limits = (min(nominal), max(nominal))
    ax.plot(limits, limits, '--', lw=2, zorder=1, color='grey', alpha=0.5)
    ax.grid(True)
    ax.set_xticks(nominal)

    # y-ticks: steps of 5, lower bound from data
    y_min = float(100.0 * matrix.min())
    y_lo = (int(y_min) // 5) * 5
    ax.set_yticks(range(y_lo, 96, 5))

    # niceify
    info = {
        'title': title,
        'ylabel': 'Observed',
        'xlabel': 'Nominal CI',
        'show_title': True,
        'show_legend': show_legend,
        'show_x': show_x,
        'stats': stats,
        'stats_suffix': '%',
    }
    niceify(ax, info)


def plotCoverage(
    summaries: EvaluationSummary | list[EvaluationSummary],
    proposals: Proposal | list[Proposal],
    labels: list[str] | None = None,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
    show_corr_rfx: bool = False,
) -> Path | None:
    if not isinstance(summaries, list):
        summaries = [summaries]
    if not isinstance(proposals, list):
        proposals = [proposals]
    if labels is None:
        labels = [''] * len(summaries)
    nrows = len(summaries)
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 6 * nrows), dpi=DPI, squeeze=False)
    axs = axs.flatten()

    for i, (summary, proposal, label) in enumerate(zip(summaries, proposals, labels)):
        names = (
            getNames('ffx', proposal.d)
            + getNames('sigmas', proposal.q, has_sigma_eps=proposal.has_sigma_eps)
            + (getCorrRfxNames(proposal.q) if show_corr_rfx and proposal.d_corr > 0 else [])
            + getNames('rfx', proposal.q)
        )
        stats = {
            'ECE': 100 * dictMeanExcl(summary.ece),
            'EACE': 100 * dictMeanExcl(summary.eace),
        }
        _plotCoverage(
            axs[i],
            summary.coverage,
            names,
            stats,
            title=label,
            show_legend=(i == 0),
            show_x=(i == nrows - 1),
            show_corr_rfx=show_corr_rfx,
        )
        axs[i].set_box_aspect(1)

    fig.tight_layout()

    # store
    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'coverage', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
