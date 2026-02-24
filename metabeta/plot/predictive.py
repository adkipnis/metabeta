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


def _ppDensity(
    ax: Axes,
    y_obs_tensor: torch.Tensor,  # (m, n)
    y_rep_tensor: torch.Tensor,  # (m, n, s)
    mask_n_tensor: torch.Tensor,  # (m, n)
    n_use: int = 64, # number of samples to use for y_rep KDEs
    color: str = 'darkgreen',
    left: bool = True,
) -> None:
    assert n_use <= y_rep_tensor.shape[-1], 'not enough samples'
    mask_n = toNumpy(mask_n_tensor)
    y_obs = toNumpy(y_obs_tensor)[mask_n]
    y_rep = toNumpy(y_rep_tensor)[mask_n]

    # shared grid from observed + replicated support
    pooled = np.concatenate([y_obs, y_rep.reshape(-1)])
    lo, hi = np.nanmin(pooled), np.nanmax(pooled)
    span = max(hi - lo, 1e-6)
    pad = 0.08 * span
    x = np.linspace(lo - pad, hi + pad, 256)

    # observed KDE
    obs_kde = gaussian_kde(y_obs)(x)

    # replicated KDEs (subset)
    s = y_rep.shape[1]
    idx = np.linspace(0, s - 1, num=n_use, dtype=int)
    rep_kdes = [gaussian_kde(y_rep[..., j])(x) for j in idx]
    rep_kdes_np = np.stack(rep_kdes, axis=0)
    q_lo, q_md, q_hi = np.quantile(rep_kdes_np, [0.025, 0.50, 0.975], axis=0)

    # plot PPC density envelope + observed
    ax.fill_between(x, q_lo, q_hi, color=color, alpha=0.20, label=r'$y_{rep}$ (95%)')
    ax.plot(x, q_md, color=color, lw=2.0, alpha=0.95, label=r'$y_{rep}$ (median)')
    ax.plot(x, obs_kde, color='black', lw=2.5, label=r'$y_{obs}$')
    info = {
        'xlabel': r'$y$',
        'ylabel': 'KDE',
        'despine': True,
        'show_legend': left,
        'legend_loc': 'best',
        'show_y': left,
        'ticks_ls': 0,
    }
    niceify(ax, info)


def plotPPD(
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
