from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.sbc import getFractionalRanks, simultaneousBands
from metabeta.utils.evaluation import getAllNames, getMasks, Proposal, joinSigmas
from metabeta.utils.plot import savePlot


def _plotSbcEcdf(
    ax: Axes,
    ranks: torch.Tensor,
    names: list[str],
    mask: torch.Tensor | None,
    diff: bool = False,
    upper: bool = True,
    lower: bool = True,
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
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(axis='both', labelsize=18)
    prefix = r'$\Delta$ ' if diff else ''
    ax.set_ylabel(f'{prefix}ECDF', fontsize=26, labelpad=10)
    if upper:
        ax.set_title('SBC', fontsize=28, pad=15)
        ax.legend(fontsize=18, markerscale=2.5, loc='upper left')
    if lower:
        ax.set_xlabel('fractional rank', fontsize=26, labelpad=10)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelcolor='w', size=1)


def plotSBC(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    diff: bool = True,
    plot_dir: Path | None = None,
    epoch: int | None = None,
) -> None:
    ranks = getFractionalRanks(proposal, data)
    ranks['sigmas'] = joinSigmas(ranks)
    names = getAllNames(proposal.d, proposal.q)
    masks = getMasks(data)

    # layered plot with conservative global simultaneous bands
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    n_eff_min = len(data['X'])
    for k in ('ffx', 'sigmas', 'rfx'):
        _plotSbcEcdf(ax, ranks[k], names[k], masks[k], diff=diff)
        # update n_eff_min
        mask_k = masks[k]
        if mask_k is not None:
            dims = tuple(range(mask_k.dim() - 1))
            n_eff = int(mask_k.sum(dims).min())
            n_eff_min = min(n_eff_min, n_eff)
    p, low, high = simultaneousBands(n_eff=n_eff_min, diff=diff)
    ax.fill_between(p, low, high, color='grey', alpha=0.1)
    ax.set_box_aspect(1)

    # store
    if plot_dir is not None:
        savePlot(plot_dir, 'sbc', epoch=epoch)
    plt.show()
    plt.close(fig)
