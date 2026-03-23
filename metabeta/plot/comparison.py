from pathlib import Path
import torch
from matplotlib import pyplot as plt

from metabeta.utils.evaluation import EvaluationSummary, getNames, Proposal, dictMean
from metabeta.utils.plot import savePlot
from metabeta.plot.recovery import _prepareRecoveryData, _plotRecoveryGrouped
from metabeta.plot.coverage import _plotCoverage
from metabeta.plot.sbc import _plotSbcRow

_COL_TITLES = ['Fixed Effects', 'Variances', 'Random Effects', 'Observed CI', 'Uniform ECDF']


def plotComparison(
    summaries: list[EvaluationSummary],
    proposals: list[Proposal],
    labels: list[str],
    data: dict[str, torch.Tensor],
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    nrows = len(summaries)
    fig, axs = plt.subplots(nrows, 5, figsize=(30, 6 * nrows), dpi=300, squeeze=False)

    for i, (summary, proposal, label) in enumerate(zip(summaries, proposals, labels)):
        upper = i == 0
        lower = i == nrows - 1

        # cols 0-2: recovery scatter
        targets, estimates, masks, names, metrics = _prepareRecoveryData(summary, data)
        _plotRecoveryGrouped(
            axs[i, :3], # type: ignore
            targets=targets,
            estimates=estimates,
            masks=masks,
            metrics=metrics,
            names=names,
            titles=_COL_TITLES[:3],
            ylabel=label,
            upper=upper,
            lower=lower,
        )

        # col 3: coverage
        names_cov = (
            getNames('ffx', proposal.d) +
            getNames('sigmas', proposal.q) +
            getNames('rfx', proposal.q)
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
            title=_COL_TITLES[3] if upper else None,
            show_legend=upper,
            show_x=lower,
        )
        axs[i, 3].set_ylabel('')

        # col 4: SBC
        _plotSbcRow(
            axs[i, 4],
            proposal,
            data,
            diff=False,
            title=_COL_TITLES[4] if upper else None,
            show_legend=upper,
            show_x=lower,
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
