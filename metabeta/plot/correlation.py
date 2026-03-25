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
    outside = (corr_mean < q025) | (corr_mean > q975)
    out_pct = 100.0 * float(outside.mean())

    lim = 1.02
    xgrid = np.linspace(-1, 1, 401)
    ylow = np.interp(xgrid, corr_true, q025)
    yhigh = np.interp(xgrid, corr_true, q975)

    mask_unc = eta_pairs == 0
    mask_cor = eta_pairs > 0
    ax.fill_between(xgrid, ylow, yhigh, color='grey', alpha=0.15, label='95% envelope')
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
            'title': 'RFX Correlation',
            'xlabel': 'Ground Truth',
            'ylabel': ylabel,
            'show_title': show_title,
            'show_legend': False,
            'show_x': show_x,
            'stats': {'out': out_pct},
            'stats_suffix': '%',
            'stats_loc_x': 0.83,
            'stats_loc_y': 0.05,
            'stats_box': True,
        },
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc='upper left',
            fontsize=18,
            markerscale=2.5,
        )


def plotRfxCorrelationRecovery(
    proposals: Proposal | list[Proposal],
    data: dict[str, torch.Tensor],
    labels: list[str] | None = None,
    n_sim: int = 2000,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    """Plot RFX correlation recovery against empirical envelopes."""
    if not isinstance(proposals, list):
        proposals = [proposals]
    if labels is None:
        labels = [''] * len(proposals)
    nrows = len(proposals)
    fig, axs = plt.subplots(nrows, 1, figsize=(9, 6 * nrows), dpi=300, squeeze=False)

    for i, (proposal, label) in enumerate(zip(proposals, labels)):
        upper = i == 0
        lower = i == nrows - 1
        results = evaluateCorrelation(proposal.rfx, data, n_sim=n_sim)
        _plotRecoveryWithBounds(
            axs[i, 0],
            results,
            show_title=upper,
            show_x=lower,
            ylabel=(label if label else 'Posterior mean'),
        )

    fig.tight_layout()

    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'rfx_correlation_recovery', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
