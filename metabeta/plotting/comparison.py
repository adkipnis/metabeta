from pathlib import Path
import torch
from matplotlib import pyplot as plt

from metabeta.utils.evaluation import EvaluationSummary, getNames, getCorrRfxNames, Proposal, dictMean
from metabeta.utils.plot import DPI, savePlot
from metabeta.plotting.recovery import _prepareRecoveryData, _plotRecoveryGrouped
from metabeta.plotting.coverage import _plotCoverage
from metabeta.plotting.sbc import _plotSbcRow

_COL_TITLES = [
    'Fixed Effects',
    'Variances',
    'Random Effects',
    'Observed CI',
    'Uniform ECDF',
]
_COL_TITLES_CORR = [
    'Fixed Effects',
    'Variances',
    'Correlations',
    'Random Effects',
    'Observed CI',
    'Uniform ECDF',
]


def plotComparison(
    summaries: list[EvaluationSummary],
    proposals: list[Proposal],
    labels: list[str],
    data: dict[str, torch.Tensor],
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
    show_corr_rfx: bool = False,
) -> Path | None:
    col_titles = _COL_TITLES
    ncols = 5
    nrows = len(summaries)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=DPI, squeeze=False)

    for i, (summary, proposal, label) in enumerate(zip(summaries, proposals, labels)):
        upper = i == 0
        lower = i == nrows - 1

        # cols 0-(n_rec-1): recovery scatter
        targets, estimates, masks, names, metrics = _prepareRecoveryData(summary, data)
        _plotRecoveryGrouped(
            axs[i, :3],  # type: ignore
            targets=targets,
            estimates=estimates,
            masks=masks,
            metrics=metrics,
            names=names,
            titles=col_titles[:3],
            ylabel=label,
            upper=upper,
            lower=lower,
        )

        # col n_rec: coverage
        names_cov = (
            getNames('ffx', proposal.d)
            + getNames('sigmas', proposal.q, has_sigma_eps=proposal.has_sigma_eps)
            + (getCorrRfxNames(proposal.q) if show_corr_rfx and proposal.d_corr > 0 else [])
            + getNames('rfx', proposal.q)
        )
        stats_cov = {
            'ECE': 100 * dictMean(summary.ece),
            'LCR': 100 * dictMean(summary.lcr),
        }
        _plotCoverage(
            axs[i, 3],
            summary.coverage,
            names_cov,
            stats_cov,
            title=col_titles[3] if upper else None,
            show_legend=upper,
            show_x=lower,
            show_corr_rfx=show_corr_rfx,
        )
        axs[i, 3].set_ylabel('')

        # col n_rec+1: SBC
        _plotSbcRow(
            axs[i, 4],
            proposal,
            data,
            diff=False,
            title=col_titles[4] if upper else None,
            show_legend=upper,
            show_x=lower,
            show_corr_rfx=show_corr_rfx,
        )
        axs[i, 4].set_ylabel('')

    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout()

    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'comparison', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
