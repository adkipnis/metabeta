from typing import Sequence
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import torch
from torch import distributions as D
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.predictive import (
    posteriorPredictiveSample,
    ppcICC,
    ppcWithinGroupSD,
    ppcBetweenGroupSD,
    intervalCheck,
)
from metabeta.utils.plot import niceify, savePlot


def toNumpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _PPCintervals(
    ax: Axes,
    t_obs_tensor: torch.Tensor,  # (b, m, n)
    res: dict[str, torch.Tensor],
    ylabel: str,
    legend: bool = True,
) -> None:
    # prepare
    t_obs = toNumpy(t_obs_tensor)
    lo = toNumpy(res['lo'])
    hi = toNumpy(res['hi'])
    outside = toNumpy(res['outside'])
    stats = {'Outside': res['outside_rate'].item() * 100}

    # plot ordered T_obs and PPI
    order = np.argsort(t_obs)
    x = np.arange(len(t_obs))
    ax.scatter(
        x,
        t_obs[order],
        label=r'$T_{obs}$',
        c=np.where(outside[order], 'crimson', 'black'),
        s=14,
        zorder=3,
    )
    ax.vlines(x, lo[order], hi[order], label=r'$T_{rep}$ (95%)', color='0.6', lw=1.5, alpha=0.9)

    info = {
        'xlabel': 'Dataset Index (sorted)',
        'ylabel': ylabel,
        'despine': True,
        'stats': stats,
        'stats_suffix': '%',
        'show_legend': legend,
    }
    niceify(ax, info)


def plotPPC(
    pp: D.Normal,
    data: dict[str, torch.Tensor],
    plot_dir: Path | None = None,
    epoch: int | None = None,
) -> None:
    y_obs = data['y'].unsqueeze(-1)
    y_rep = posteriorPredictiveSample(pp, data)
    fig, axs = plt.subplots(figsize=(6 * 3, 6), ncols=3, dpi=300)

    # within group SD
    sd_within_rep = ppcWithinGroupSD(y_rep, data)
    sd_within_obs = ppcWithinGroupSD(y_obs, data).squeeze(-1)
    sd_within_res = intervalCheck(sd_within_obs, sd_within_rep)
    _PPCintervals(axs[0], sd_within_obs, sd_within_res, 'Within Group SD', legend=True)

    # between group SD
    sd_between_rep = ppcBetweenGroupSD(y_rep, data)
    sd_between_obs = ppcBetweenGroupSD(y_obs, data).squeeze(-1)
    sd_between_res = intervalCheck(sd_between_obs, sd_between_rep)
    _PPCintervals(axs[1], sd_between_obs, sd_between_res, 'Between Group SD', legend=False)

    # ICC
    icc_rep = ppcICC(sd_between_rep, sd_within_rep)
    icc_obs = ppcICC(sd_between_obs, sd_within_obs)
    icc_res = intervalCheck(icc_obs, icc_rep)
    _PPCintervals(axs[2], icc_obs, icc_res, 'Intra-Class Correlation', legend=False)

    # format
    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout()

    # store
    if plot_dir is not None:
        savePlot(plot_dir, 'posterior_predictive_checks', epoch=epoch)
    plt.show()
    plt.close(fig)
