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


def pointwiseBands(
    n_eff: int,  # effective number of posteriors
    alpha: float = 0.05,  # alpha error for bounds
    diff: bool = False,  # use ECDF delta
    eps: float = 1e-5,
) -> tuple[np.ndarray, ...]:
    """Pointwise ECDF bands under SBC null (too strict)"""
    z = np.linspace(0.0 + eps, 1.0 - eps, n_eff)
    lower = binom(n_eff, z).ppf(alpha / 2.0) / n_eff
    upper = binom(n_eff, z).ppf(1.0 - alpha / 2.0) / n_eff
    if diff:
        lower -= z
        upper -= z
    return z, lower, upper


def simultaneousBands(
    n_eff: int,  # effective number of posteriors
    alpha: float = 0.05,  # alpha error for bounds
    diff: bool = False,  # use ECDF delta
    max_n: int = 1000,  # upper bound for n_eff
    n_sim: int = 2000,  # number of draws from the uniform
    eps: float = 1e-5,
) -> tuple[np.ndarray, ...]:
    """Simultanous ECDF bands as proposed by Sailynoja et al. (2022):
    - sample n_sim uniform values for each posterior
    - get minimal coverage probabilities of uniform samples
    - use the alpha quantile of these probabilites instead of exact alpha
    """
    K = min(n_eff, max_n)
    z = np.linspace(0.0 + eps, 1.0 - eps, K)
    u = np.random.uniform(size=(n_sim, n_eff))
    gammas = _minimalCoverageProbs(z, u)   # (n_sim, )
    gamma = np.percentile(gammas, 100.0 * alpha)
    lower = binom(n_eff, z).ppf(gamma / 2.0) / n_eff
    upper = binom(n_eff, z).ppf(1.0 - gamma / 2.0) / n_eff
    if diff:
        lower -= z
        upper -= z
    return z, lower, upper


def _minimalCoverageProbs(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """gamma per for n_sim simulations"""
    n_eff = u.shape[1]
    # ECDF values of each simulated sample at each z
    F_m = np.sum(z[:, None] >= u[:, None, :], axis=-1) / n_eff
    bin1 = binom(n_eff, z).cdf(n_eff * F_m)
    bin2 = binom(n_eff, z).cdf(n_eff * F_m - 1)
    gamma = 2 * np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma


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
        fname = plot_dir / 'sbc_latest.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
        if epoch is not None:
            fname_e = plot_dir / f'sbc_e{epoch}.png'
            plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15)
    plt.show()
    plt.close(fig)
