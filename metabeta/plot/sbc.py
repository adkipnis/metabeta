from pathlib import Path
import torch
import numpy as np
from scipy.stats import binom
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


from metabeta.utils.evaluation import getAllNames, getMasks, Proposal, joinSigmas


def fractionalRanks(
    samples: torch.Tensor,  # (b, ..., s, d)
    targets: torch.Tensor,  # (b, ..., d)
) -> torch.Tensor:
    sample_dim = -1 if targets.dim() == 1 else -2
    targets = targets.unsqueeze(sample_dim)
    smaller = samples < targets
    return smaller.float().mean(sample_dim)


def getFractionalRanks(
    proposal: Proposal, data: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    param_names = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
    ranks = {}
    for param in param_names:
        samples = getattr(proposal, param)
        targets = data[param]
        ranks[param] = fractionalRanks(samples, targets)
    return ranks


def theoreticalLimits(
    b: int,  # number of posteriors (batch size)
    s: int = 1000,  # number of samples per posterior
    alpha: float = 0.05,  # alpha error for bounds
    diff: bool = False,
) -> tuple:
    p = np.linspace(0, 1, b)
    lower = binom.ppf(alpha / 2, s, p) / s
    upper = binom.ppf(1 - alpha / 2, s, p) / s
    if diff:
        lower -= p
        upper -= p
    return p, lower, upper


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
            x = ranks[mask_i, i].sort()[0].numpy()
        else:
            x = ranks.view(-1, ranks.shape[-1])[:, i].sort()[0].numpy()
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
