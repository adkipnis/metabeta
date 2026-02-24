from pathlib import Path
import numpy as np
import torch
from torch import distributions as D
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.predictive import (
    posteriorPredictiveSample,
    posteriorPredictiveWithinGroupSD,
    intervalCheck,
)
from metabeta.utils.plot import niceify, savePlot


def detach(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _PPCintervals(
    ax: Axes,
    t_obs_tensor: torch.Tensor,
    res: dict[str, torch.Tensor],
    ylabel: str,
) -> None:
    # prepare
    t_obs = detach(t_obs_tensor)
    lo = detach(res['lo'])
    hi = detach(res['hi'])
    outside = detach(res['outside'])
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
    }
    niceify(ax, info)


def plotPPC(
    pp: D.Normal,
    data: dict[str, torch.Tensor],
    plot_dir: Path | None = None,
    epoch: int | None = None,
) -> None:
    fig, axs = plt.subplots(figsize=(6 * 3, 6), ncols=3, dpi=300)

    y_obs = data['y'].unsqueeze(-1)
    y_rep = posteriorPredictiveSample(pp, data)

    # within group SD
    t_rep = posteriorPredictiveWithinGroupSD(y_rep, data)
    t_obs = posteriorPredictiveWithinGroupSD(y_obs, data).squeeze(-1)
    res = intervalCheck(t_obs, t_rep)
    _PPCintervals(axs[0], t_obs, res, 'Within Group SD')
    
    # format
    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout()

    # store
    if plot_dir is not None:
        savePlot(plot_dir, 'posterior_predictive_checks', epoch=epoch)
    plt.show()
    plt.close(fig)
