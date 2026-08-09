from pathlib import Path
import torch
from matplotlib import pyplot as plt

from metabeta.utils.evaluation import (
    EvaluationSummary,
    dictMean,
)
from metabeta.utils.plot import DPI, legendProxy, savePlot
from metabeta.utils.results import getNames, getCorrRfxNames, Proposal
from metabeta.plotting.recovery import _prepareRecoveryData, _plotRecoveryGrouped
from metabeta.plotting.coverage import _plotCoverage
from metabeta.plotting.sbc import _plotSbcRow

_COL_TITLES = [
    'Fixed Effects',
    'Variances',
    'Random Effects',
    'Empirical Coverage',
    'Δ Uniform ECDF',
]
_COL_TITLES_CORR = [
    'Fixed Effects',
    'Variances',
    'Correlations',
    'Random Effects',
    'Empirical Coverage',
    'Δ Uniform ECDF',
]
_RECOVERY_ALPHA = 0.35
_LINE_ALPHA = 0.75
_RIGHT_LEGEND_FONTSIZE = 30
# Anchor of the right legend's left edge. Tuned against _PLOT_RIGHT_MARGIN so the gap to the
# outermost panel is ~40% smaller than the old 0.90 anchor; the legend still ends near 0.96,
# well inside the figure, so a larger font does not overflow.
_RIGHT_LEGEND_X = 0.890
_PLOT_RIGHT_MARGIN = 0.89


def _rightLegendHandles(axs) -> dict[str, object]:
    handles_by_label: dict[str, object] = {}
    for ax in axs.flat:
        handles, axis_labels = ax.get_legend_handles_labels()
        for handle, axis_label in zip(handles, axis_labels):
            if (
                axis_label.startswith('_')
                or axis_label.startswith('95% CB')
                or axis_label in handles_by_label
            ):
                continue
            # opaque stand-in: the plotted artists are semi-transparent, which makes the shared
            # legend's swatches too washed out to match against the panels
            handles_by_label[axis_label] = legendProxy(handle, axis_label)
    return handles_by_label


def plotComparison(
    summaries: list[EvaluationSummary],
    proposals: list[Proposal],
    labels: list[str],
    data: dict[str, torch.Tensor],
    plot_dir: Path | None = None,
    plot_name: str = 'comparison',
    epoch: int | None = None,
    show: bool = False,
    show_corr_rfx: bool = False,
    legend_right: bool = False,
) -> Path | None:
    col_titles = _COL_TITLES_CORR if show_corr_rfx else _COL_TITLES
    n_rec = 4 if show_corr_rfx else 3
    ncols = n_rec + 2
    nrows = len(summaries)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=DPI, squeeze=False)

    for i, (summary, proposal, label) in enumerate(zip(summaries, proposals, labels)):
        upper = i == 0
        lower = i == nrows - 1

        # cols 0-(n_rec-1): recovery scatter
        targets, estimates, masks, names, metrics = _prepareRecoveryData(
            summary, data, show_corr_rfx=show_corr_rfx
        )
        _plotRecoveryGrouped(
            axs[i, :n_rec],  # type: ignore
            targets=targets,
            estimates=estimates,
            masks=masks,
            metrics=metrics,
            names=names,
            titles=col_titles[:n_rec],
            ylabel=label,
            upper=upper,
            lower=lower,
            alpha=_RECOVERY_ALPHA,
            show_legend=False if legend_right else None,
        )

        # col n_rec: coverage
        names_cov = (
            getNames('ffx', proposal.d)
            + getNames('sigmas', proposal.q, has_sigma_eps=proposal.has_sigma_eps)
            + (getCorrRfxNames(proposal.q) if show_corr_rfx and proposal.d_corr > 0 else [])
            + getNames('rfx', proposal.q)
        )
        stats_cov = {
            'ECE': 100 * dictMean(summary.aggregated.ece),
            'EACE': 100 * dictMean(summary.aggregated.eace),
        }
        _plotCoverage(
            axs[i, n_rec],
            summary.aggregated.coverage,
            names_cov,
            stats_cov,
            title=col_titles[n_rec] if upper else None,
            show_legend=False,
            show_x=lower,
            show_corr_rfx=show_corr_rfx,
            line_alpha=_LINE_ALPHA,
        )
        axs[i, n_rec].set_ylabel('')

        # col n_rec+1: SBC
        _plotSbcRow(
            axs[i, n_rec + 1],
            proposal,
            data,
            diff=True,
            title=col_titles[n_rec + 1] if upper else None,
            show_legend=False,
            show_band_legend=upper,
            show_x=lower,
            show_corr_rfx=show_corr_rfx,
            draw_legend=True,
            line_alpha=_LINE_ALPHA,
        )
        axs[i, n_rec + 1].set_ylabel('')

    for ax in axs.flat:
        ax.set_box_aspect(1)
    if legend_right:
        handles_by_label = _rightLegendHandles(axs)
        if handles_by_label:
            fig.legend(
                handles_by_label.values(),
                handles_by_label.keys(),
                loc='center left',
                bbox_to_anchor=(_RIGHT_LEGEND_X, 0.5),
                fontsize=_RIGHT_LEGEND_FONTSIZE,
                markerscale=2.5,
            )
        fig.tight_layout(rect=(0.0, 0.0, _PLOT_RIGHT_MARGIN, 1.0))
    else:
        fig.tight_layout()

    saved_path = None
    if plot_dir is not None:
        savePlot(plot_dir, plot_name, epoch=epoch, ending='pdf')
        saved_path = savePlot(plot_dir, plot_name, epoch=epoch, ending='png')
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
