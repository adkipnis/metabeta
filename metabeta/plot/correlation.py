from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.correlation import evaluateCorrelation
from metabeta.utils.evaluation import Proposal
from metabeta.utils.plot import niceify, savePlot


def plotCorrelation(
    x: np.ndarray | pd.DataFrame,
    names: list[str] | None = None,
) -> None:
    """Plot correlation matrix over columns of x."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    corr = np.corrcoef(x, rowvar=False)
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='RdBu')

    cbar = fig.colorbar(
        cax,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_ticks(np.linspace(-1, 1, 5).tolist())
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(axis='x', bottom=False, top=True)
    d = len(corr)
    ticks = np.arange(0, d, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if names is None:
        names = [rf'$x_{i + 1}$' for i in range(d)]
    ax.set_xticklabels(names, fontsize=20)
    ax.set_yticklabels(names, fontsize=20)

    plt.show()


def _flattenPairs(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().reshape(-1).numpy()


def _plotRecoveryWithBounds(
    ax: Axes,
    results: dict[str, torch.Tensor],
    show_title: bool,
    show_legend: bool,
    show_x: bool,
    ylabel: str,
) -> None:
    corr_true = _flattenPairs(results['corr_true'])
    corr_mean = _flattenPairs(results['corr_mean'])
    q025 = _flattenPairs(results['corr_q025'])
    q975 = _flattenPairs(results['corr_q975'])
    eta = _flattenPairs(results['eta_rfx'])
    n_pairs = results['corr_true'].shape[-1]
    eta_pairs = np.repeat(eta, n_pairs)

    order = np.argsort(corr_true)
    corr_true = corr_true[order]
    corr_mean = corr_mean[order]
    q025 = q025[order]
    q975 = q975[order]
    eta_pairs = eta_pairs[order]

    lim = 1.02
    xgrid = np.linspace(-1, 1, 401)
    ylow = np.interp(xgrid, corr_true, q025)
    yhigh = np.interp(xgrid, corr_true, q975)

    mask_unc = eta_pairs == 0
    mask_cor = eta_pairs > 0
    ax.fill_between(xgrid, ylow, yhigh, color='grey', alpha=0.15, label='Empirical 95% envelope')
    if mask_unc.any():
        ax.scatter(
            corr_true[mask_unc],
            corr_mean[mask_unc],
            s=35,
            alpha=0.40,
            label='Uncorrelated',
        )
    if mask_cor.any():
        ax.scatter(
            corr_true[mask_cor],
            corr_mean[mask_cor],
            s=35,
            alpha=0.40,
            label='Correlated',
        )
    ax.plot([-1, 1], [-1, 1], '--', lw=2, color='grey', alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_axisbelow(True)
    ax.grid(True)
    niceify(
        ax,
        {
            'title': 'RFX Correlation Recovery',
            'xlabel': 'True correlation',
            'ylabel': ylabel,
            'show_title': show_title,
            'show_legend': show_legend,
            'show_x': show_x,
        },
    )


def _plotDetectionSeparation(
    ax: Axes,
    results: dict[str, torch.Tensor],
    threshold: float,
    show_title: bool,
    show_legend: bool,
    show_x: bool,
) -> None:
    eta = _flattenPairs(results['eta_rfx'])
    n_pairs = results['corr_true'].shape[-1]
    eta_pairs = np.repeat(eta, n_pairs)
    percentile_pairs = _flattenPairs(results['percentile_pairs'])

    unc = np.sort(percentile_pairs[eta_pairs == 0])
    cor = np.sort(percentile_pairs[eta_pairs > 0])
    if unc.size > 0:
        y_unc = np.linspace(0, 1, unc.size)
        ax.plot(unc, y_unc, lw=3, label='Uncorrelated')
    if cor.size > 0:
        y_cor = np.linspace(0, 1, cor.size)
        ax.plot(cor, y_cor, lw=3, label='Correlated')

    tpr = float((cor > threshold).mean()) if cor.size > 0 else float('nan')
    fpr = float((unc > threshold).mean()) if unc.size > 0 else float('nan')

    ax.axvline(threshold, linestyle='--', lw=2, color='grey', alpha=0.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_axisbelow(True)
    ax.grid(True)
    niceify(
        ax,
        {
            'title': 'Detection Separation',
            'xlabel': 'Null percentile of |r|',
            'ylabel': 'ECDF',
            'show_title': show_title,
            'show_legend': show_legend,
            'show_x': show_x,
            'stats': {'TPR': tpr, 'FPR': fpr},
        },
    )


def plotRfxCorrelationRecovery(
    proposals: Proposal | list[Proposal],
    data: dict[str, torch.Tensor],
    labels: list[str] | None = None,
    threshold: float = 0.90,
    n_sim: int = 2000,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    """Plot RFX correlation recovery and detection separation."""
    if not isinstance(proposals, list):
        proposals = [proposals]
    if labels is None:
        labels = [''] * len(proposals)
    nrows = len(proposals)
    fig, axs = plt.subplots(nrows, 2, figsize=(12, 6 * nrows), dpi=300, squeeze=False)

    for i, (proposal, label) in enumerate(zip(proposals, labels)):
        upper = i == 0
        lower = i == nrows - 1
        results = evaluateCorrelation(proposal.rfx, data, n_sim=n_sim)
        _plotRecoveryWithBounds(
            axs[i, 0],
            results,
            show_title=upper,
            show_legend=upper,
            show_x=lower,
            ylabel=(label if label else 'Posterior mean correlation'),
        )
        _plotDetectionSeparation(
            axs[i, 1],
            results,
            threshold=threshold,
            show_title=upper,
            show_legend=upper,
            show_x=lower,
        )

    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout()

    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'rfx_correlation_recovery', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
