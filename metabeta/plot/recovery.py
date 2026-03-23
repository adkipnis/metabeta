from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from metabeta.utils.evaluation import EvaluationSummary, getMasks, getNames, joinSigmas
from metabeta.utils.plot import PALETTE, savePlot, niceify


def _plotRecovery(
    ax: Axes,
    targets: np.ndarray,
    estimates: np.ndarray,
    mask: np.ndarray,
    stats: dict[str, float],
    # strings
    names: list[str],
    colors: list[str],
    title: str = '',
    ylabel: str = 'Estimate',
    # plot details
    marker: str = 'o',
    alpha: float = 0.15,
    upper: bool = True,
    lower: bool = True,
) -> None:
    """scatter plot: ground truth vs. estimates"""
    # check sizes
    d = len(names)
    assert targets.shape[-1] == estimates.shape[-1] == d, 'shape mismatch'

    # init figure
    ax.set_axisbelow(True)
    ax.grid(True)
    min_val = min(targets.min(), estimates.min())
    max_val = max(targets.max(), estimates.max())
    delta = max_val - min_val
    const = delta * 0.05
    limits = (min_val - const, max_val + const)
    ax.set_xlim(limits, auto=False)
    ax.set_ylim(limits, auto=False)
    ax.plot(limits, limits, '--', lw=2, zorder=1, color='grey', alpha=0.5)  # diagline

    # overlay plots
    for i in range(d):
        mask_i = mask[..., i]
        if mask_i.sum() == 0:
            continue
        targets_i = targets[mask_i, i]
        estimates_i = estimates[mask_i, i]
        color_i = colors[i]
        ax.scatter(
            targets_i,
            estimates_i,
            color=color_i,
            s=70,
            marker=marker,
            alpha=alpha,
            label=names[i],
        )

    # final touches
    ml = max(np.floor(delta / 4), 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(ml))
    ax.yaxis.set_major_locator(MultipleLocator(ml))
    info = {
        'title': title,
        'ylabel': ylabel,
        'xlabel': 'Ground Truth',
        'show_title': upper,
        'show_legend': upper,
        'show_x': lower,
        'stats': stats,
    }
    niceify(ax, info)


def _plotRecoveryGrouped(
    axs: Axes,
    targets: list[torch.Tensor],
    estimates: list[torch.Tensor],
    masks: list[torch.Tensor],
    metrics: list[dict[str, float]],
    # string
    names: list[list[str]],
    titles: list[str] = [],
    ylabel: str = 'Estiamte',
    # plot details
    marker: str = 'o',
    alpha: float = 0.25,
    upper: bool = True,
    lower: bool = True,
) -> None:
    i = 0
    for ax, tar, est, mas, met, nam, tit in zip(
        axs,
        targets,
        estimates,
        masks,
        metrics,
        names,
        titles,  # type: ignore
    ):
        col = PALETTE[i : i + len(nam)]
        _plotRecovery(
            ax,
            targets=tar.numpy(),
            estimates=est.numpy(),
            mask=mas.numpy(),
            stats=met,
            names=nam,
            colors=col,
            title=tit,
            ylabel=(ylabel if i == 0 else ''),
            marker=marker,
            alpha=alpha,
            upper=upper,
            lower=lower,
        )
        i += len(nam)


def _prepareRecoveryData(
    summary: EvaluationSummary,
    data: dict[str, torch.Tensor],
) -> tuple[list, list, list, list, list]:
    allMasks = getMasks(data)
    est: dict[str, torch.Tensor] = summary.estimates
    stats = {'corr': summary.corr, 'nrmse': summary.nrmse}

    targets, estimates, masks, names, metrics = [], [], [], [], []

    # fixed effects
    d = data['ffx'].shape[-1]
    targets.append(data['ffx'])
    estimates.append(est['ffx'])
    masks.append(allMasks['ffx'])
    names.append(getNames('ffx', d))
    metrics.append(
        {
            'r': stats['corr']['ffx'].mean().item(),
            'NRMSE': stats['nrmse']['ffx'].mean().item(),
        }
    )

    # variance parameters
    q = data['sigma_rfx'].shape[-1]
    targets.append(joinSigmas(data))
    estimates.append(joinSigmas(est))
    masks.append(allMasks['sigmas'])
    names.append(getNames('sigmas', q))
    nrmse = q / (q + 1) * stats['nrmse']['sigma_rfx'] + 1 / (q + 1) * stats['nrmse']['sigma_eps']
    r = q / (q + 1) * stats['corr']['sigma_rfx'] + 1 / (q + 1) * stats['corr']['sigma_eps']
    metrics.append(
        {
            'r': r.mean().item(),
            'NRMSE': nrmse.mean().item(),
        }
    )

    # random effects
    targets.append(data['rfx'].view(-1, q))
    estimates.append(est['rfx'].view(-1, q))
    masks.append(allMasks['rfx'].view(-1, q))  # type: ignore
    names.append(getNames('rfx', q))
    metrics.append(
        {
            'r': stats['corr']['rfx'].mean().item(),
            'NRMSE': stats['nrmse']['rfx'].mean().item(),
        }
    )

    return targets, estimates, masks, names, metrics


def plotRecovery(
    summaries: EvaluationSummary | list[EvaluationSummary],
    data: dict[str, torch.Tensor],
    labels: list[str] | None = None,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    if not isinstance(summaries, list):
        summaries = [summaries]
    if labels is None:
        labels = [None] * len(summaries)  # type: ignore[list-item]
    nrows = len(summaries)
    fig, axs = plt.subplots(nrows, 3, figsize=(18, 6 * nrows), dpi=300, squeeze=False)

    for i, (summary, label) in enumerate(zip(summaries, labels)):
        upper = i == 0
        lower = i == nrows - 1
        targets, estimates, masks, names, metrics = _prepareRecoveryData(summary, data)
        _plotRecoveryGrouped(
            axs[i],
            targets=targets,
            estimates=estimates,
            masks=masks,
            metrics=metrics,
            names=names,
            titles=['Fixed Effects', 'Variances', 'Random Effects'],
            ylabel=label,
            upper=upper,
            lower=lower,
        )

    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout()

    # store
    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'parameter_recovery', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
