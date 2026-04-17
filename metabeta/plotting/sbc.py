from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.sbc import getFractionalRanks, simultaneousBands
from metabeta.utils.evaluation import getAllNames, getCorrRfxNames, getMasks, Proposal, joinSigmas
from metabeta.utils.plot import DPI, savePlot, niceify


def _plotSbcEcdf(
    ax: Axes,
    ranks: torch.Tensor,
    names: list[str],
    mask: torch.Tensor | None,
    diff: bool,
    title: str | None = 'Uniform ECDF',
    show_legend: bool = True,
    show_x: bool = True,
) -> None:
    assert len(names) == ranks.shape[-1], 'shape mismatch'
    for i, name in enumerate(names):
        if mask is not None:
            mask_i = mask[..., i]
            x = ranks[mask_i, i].sort()[0]
        else:
            x = ranks.view(-1, ranks.shape[-1])[:, i].sort()[0]
        if x.numel() == 0:
            continue
        x = x.detach().cpu().numpy()
        x = np.pad(x, (1, 1), constant_values=(0, 1))
        y = np.linspace(0, 1, num=x.shape[-1])
        if diff:
            y = y - x
        ax.plot(x, y, label=name, lw=3)
    xlim = (0, 1)
    ylim = (0, 0) if diff else xlim
    ax.plot(xlim, ylim, '--', lw=2, zorder=1, color='grey', alpha=0.2)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.02)

    # niceify
    prefix = r'$\Delta$ ' if diff else ''
    info = {
        'title': title,
        'ylabel': f'{prefix}Uniform ECDF',
        'xlabel': 'SBC Fractional Rank',
        'show_title': True,
        'show_legend': show_legend,
        'show_x': show_x,
    }
    niceify(ax, info)


def _nEff(mask: torch.Tensor | None, n_datasets: int) -> int:
    if mask is None:
        return n_datasets
    dims = tuple(range(mask.dim() - 1))
    return int(mask.sum(dims).min())


def _plotSbcRow(
    ax: Axes,
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    diff: bool,
    title: str | None,
    show_legend: bool,
    show_x: bool,
) -> None:
    ranks = getFractionalRanks(proposal, data)
    ranks['sigmas'] = joinSigmas(ranks)
    has_eps = proposal.has_sigma_eps
    names = getAllNames(proposal.d, proposal.q, has_sigma_eps=has_eps)
    masks = getMasks(data, has_sigma_eps=has_eps)

    n_datasets = len(data['X'])

    # Global parameters: ffx, sigmas, corr_rfx
    n_eff_global = n_datasets
    for k in ('ffx', 'sigmas'):
        _plotSbcEcdf(
            ax,
            ranks[k],
            names[k],
            masks[k],
            diff=diff,
            title=title,
            show_legend=show_legend,
            show_x=show_x,
        )
        n_eff_global = min(n_eff_global, _nEff(masks[k], n_datasets))
    if 'corr_rfx' in ranks:
        _plotSbcEcdf(
            ax,
            ranks['corr_rfx'],
            getCorrRfxNames(proposal.q),
            mask=None,
            diff=diff,
            title=title,
            show_legend=show_legend,
            show_x=show_x,
        )
        # corr_rfx only present for q > 1 datasets; no per-entry mask
        n_eff_global = min(n_eff_global, n_datasets)
    p, low, high = simultaneousBands(n_eff=n_eff_global, diff=diff)
    ax.fill_between(p, low, high, color='grey', alpha=0.2, label='95% band (global)')

    # Local parameters: rfx
    _plotSbcEcdf(
        ax,
        ranks['rfx'],
        names['rfx'],
        masks['rfx'],
        diff=diff,
        title=title,
        show_legend=show_legend,
        show_x=show_x,
    )
    n_eff_local = _nEff(masks['rfx'], n_datasets)
    p, low, high = simultaneousBands(n_eff=n_eff_local, diff=diff)
    ax.fill_between(p, low, high, color='steelblue', alpha=0.1, label='95% band (local)')

    if show_legend:
        ax.legend(fontsize=18, markerscale=2.5, loc='upper left')


def plotSBC(
    proposals: Proposal | list[Proposal],
    data: dict[str, torch.Tensor],
    labels: list[str] | None = None,
    diff: bool = False,
    plot_dir: Path | None = None,
    epoch: int | None = None,
    show: bool = False,
) -> Path | None:
    if not isinstance(proposals, list):
        proposals = [proposals]
    if labels is None:
        labels = [''] * len(proposals)
    nrows = len(proposals)
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 6 * nrows), dpi=DPI, squeeze=False)
    axs = axs.flatten()

    for i, (proposal, label) in enumerate(zip(proposals, labels)):
        _plotSbcRow(
            axs[i],
            proposal,
            data,
            diff=diff,
            title=label,
            show_legend=(i == 0),
            show_x=(i == nrows - 1),
        )
        axs[i].set_box_aspect(1)

    fig.tight_layout()

    # store
    saved_path = None
    if plot_dir is not None:
        saved_path = savePlot(plot_dir, 'sbc', epoch=epoch)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
