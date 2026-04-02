from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.correlation import evaluateCorrelation
from metabeta.utils.evaluation import Proposal
from metabeta.utils.plot import DPI, niceify, savePlot


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


def _sizeEncoding(
    m_pairs: np.ndarray,
    size_range: tuple[float, float],
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    m_min, m_max = float(m_pairs.min()), float(m_pairs.max())
    if m_max > m_min:
        sizes = np.interp(m_pairs, (m_min, m_max), size_range)
        m_levels = np.unique(np.round(np.quantile(m_pairs, [0.0, 0.5, 1.0])).astype(int))
        size_labels = [
            (int(m), float(np.interp(float(m), (m_min, m_max), size_range))) for m in m_levels
        ]
    else:
        sizes = np.full_like(m_pairs, np.mean(size_range), dtype=float)
        size_labels = [(int(m_min), float(np.mean(size_range)))]
    return sizes, size_labels


def _plotRecoveryWithBounds(
    ax: Axes,
    results: dict[str, torch.Tensor],
    show_title: bool,
    show_x: bool,
    ylabel: str,
    size_range: tuple[float, float] = (20.0, 140.0),
) -> None:
    corr_true = _flattenPairs(results['corr_true'])
    corr_mean = _flattenPairs(results['corr_mean'])
    q025 = _flattenPairs(results['corr_q025'])
    q975 = _flattenPairs(results['corr_q975'])
    post_q025 = _flattenPairs(results['post_q025'])
    post_q975 = _flattenPairs(results['post_q975'])
    eta_pairs = _flattenPairs(results['eta_rfx_pairs'])
    m_pairs = _flattenPairs(results['m_pairs'])

    order = np.argsort(corr_true)
    corr_true = corr_true[order]
    corr_mean = corr_mean[order]
    q025 = q025[order]
    q975 = q975[order]
    post_q025 = post_q025[order]
    post_q975 = post_q975[order]
    eta_pairs = eta_pairs[order]
    m_pairs = m_pairs[order]

    sizes, size_labels = _sizeEncoding(m_pairs, size_range)

    outside = (corr_true < post_q025) | (corr_true > post_q975)
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
            s=sizes[mask_unc],
            alpha=0.40,
            label='Uncorrelated',
        )
    if mask_cor.any():
        ax.scatter(
            corr_true[mask_cor],
            corr_mean[mask_cor],
            s=sizes[mask_cor],
            alpha=0.40,
            label='Correlated',
        )
    ax.plot([-1, 1], [-1, 1], '--', lw=2, color='grey', alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ticks = np.arange(-1.0, 1.0001, 0.25)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
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
            'stats_loc_x': 0.81,
            'stats_loc_y': 0.05,
            'stats_box': True,
        },
    )
    if show_title:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            class_legend = ax.legend(
                handles,
                labels,
                loc='upper left',
                fontsize=18,
            )
            ax.add_artist(class_legend)
        if size_labels:
            size_handles = [
                ax.scatter([], [], s=s_level, alpha=0.40, color='grey')
                for _, s_level in size_labels
            ]
            size_text = [f'{m_level}' for m_level, _ in size_labels]
            ax.legend(
                size_handles,
                size_text,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=18,
                title='Groups',
                title_fontsize=18,
            )


def _plotSampleVsPosterior(
    ax: Axes,
    results: dict[str, torch.Tensor],
    show_title: bool,
    show_x: bool,
    ylabel: str,
    size_range: tuple[float, float] = (20.0, 140.0),
) -> None:
    corr_sample = _flattenPairs(results['corr_sample'])
    corr_mean = _flattenPairs(results['corr_mean'])
    eta_pairs = _flattenPairs(results['eta_rfx_pairs'])
    m_pairs = _flattenPairs(results['m_pairs'])

    order = np.argsort(corr_sample)
    corr_sample = corr_sample[order]
    corr_mean = corr_mean[order]
    eta_pairs = eta_pairs[order]
    m_pairs = m_pairs[order]

    sizes, _ = _sizeEncoding(m_pairs, size_range)

    mae = float(np.abs(corr_mean - corr_sample).mean())

    lim = 1.02
    mask_unc = eta_pairs == 0
    mask_cor = eta_pairs > 0
    if mask_unc.any():
        ax.scatter(
            corr_sample[mask_unc],
            corr_mean[mask_unc],
            s=sizes[mask_unc],
            alpha=0.40,
            label='Uncorrelated',
        )
    if mask_cor.any():
        ax.scatter(
            corr_sample[mask_cor],
            corr_mean[mask_cor],
            s=sizes[mask_cor],
            alpha=0.40,
            label='Correlated',
        )
    ax.plot([-1, 1], [-1, 1], '--', lw=2, color='grey', alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ticks = np.arange(-1.0, 1.0001, 0.25)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_axisbelow(True)
    ax.grid(True)
    niceify(
        ax,
        {
            'title': 'Sample vs Posterior',
            'xlabel': 'Sample r',
            'ylabel': ylabel,
            'show_title': show_title,
            'show_legend': False,
            'show_x': show_x,
            'stats': {'MAE': mae},
            'stats_loc_x': 0.81,
            'stats_loc_y': 0.05,
            'stats_box': True,
        },
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles and show_title:
        ax.legend(handles, labels, loc='upper left', fontsize=18)


def plotRfxCorrelationFromResults(
    results_list: list[dict],
    labels: list[str] | None = None,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    """Plot sample-vs-posterior rfx correlation from pre-computed results dicts.

    Accepts output of evaluateCorrelation (or mergeCorrelationResults), so callers
    can aggregate results across multiple configs before plotting.
    """
    if labels is None:
        labels = [''] * len(results_list)
    nrows = len(results_list)
    fig, axs = plt.subplots(nrows, 1, figsize=(11, 6 * nrows), dpi=DPI, squeeze=False)

    for i, (results, label) in enumerate(zip(results_list, labels)):
        _plotRecoveryWithBounds(
            axs[i, 0],
            results,
            show_title=(i == 0),
            show_x=(i == nrows - 1),
            ylabel=(label if label else 'Posterior mean'),
        )

    fig.tight_layout(rect=(0, 0, 0.86, 1))

    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'rfx_correlation_recovery', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def plotRfxCorrelationRecovery(
    proposals: Proposal | list[Proposal],
    data: dict[str, torch.Tensor],
    labels: list[str] | None = None,
    n_sim: int = 2000,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    """Evaluate and plot sample-vs-posterior rfx correlation recovery."""
    if not isinstance(proposals, list):
        proposals = [proposals]
    results_list = [evaluateCorrelation(p.rfx, data, n_sim=n_sim) for p in proposals]
    return plotRfxCorrelationFromResults(
        results_list, labels=labels, plot_dir=plot_dir, epoch=epoch, show=show
    )
