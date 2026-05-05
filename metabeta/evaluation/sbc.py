import torch
import numpy as np
from scipy.stats import binom, beta as beta_dist

from metabeta.utils.evaluation import Proposal
from metabeta.utils.regularization import corrToLower


def fractionalRanks(
    samples: torch.Tensor,  # (b, ..., s, d)
    targets: torch.Tensor,  # (b, ..., d)
    weights: torch.Tensor | None = None,  # (b, s)
) -> torch.Tensor:
    sample_dim = -1 if targets.dim() == 1 else -2
    targets = targets.unsqueeze(sample_dim)
    smaller = (samples < targets).float()
    if weights is None:
        return smaller.mean(sample_dim)
    if sample_dim == -2:
        weights = weights.unsqueeze(-1)
    if targets.dim() == 4:
        weights = weights.unsqueeze(1)
    return (smaller * weights).sum(sample_dim)


def getFractionalRanks(
    proposal: Proposal, data: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    param_names = ['ffx', 'sigma_rfx', 'rfx']
    if proposal.has_sigma_eps:
        param_names.insert(2, 'sigma_eps')
    weights = proposal.weights
    ranks = {}
    for param in param_names:
        samples = getattr(proposal, param)
        targets = data[param]
        ranks[param] = fractionalRanks(samples, targets, weights)
    if proposal.corr_rfx is not None:
        ds_mask = data['mask_q'][:, 1]  # True only for q>=2 datasets
        if ds_mask.any():
            ranks['corr_rfx'] = fractionalRanks(
                corrToLower(proposal.corr_rfx)[ds_mask],
                corrToLower(data['corr_rfx'])[ds_mask],
                weights[ds_mask] if weights is not None else None,
            )
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
    smooth: bool = False,  # use beta distribution for smooth bands
) -> tuple[np.ndarray, ...]:
    """Simultanous ECDF bands as proposed by Sailynoja et al. (2022):
    - sample n_sim uniform values for each posterior
    - get minimal coverage probabilities of uniform samples
    - use the alpha quantile of these probabilites instead of exact alpha
    """
    K = min(n_eff, max_n)
    z_sim = np.linspace(0.0 + eps, 1.0 - eps, K)
    u = np.random.uniform(size=(n_sim, n_eff))
    gammas = _minimalCoverageProbs(z_sim, u)   # (n_sim, )
    gamma = np.percentile(gammas, 100.0 * alpha)
    if smooth:
        z = np.linspace(0.0 + eps, 1.0 - eps, max(K, 500))
        lower = beta_dist.ppf(gamma / 2.0, n_eff * z, n_eff * (1 - z) + 1)
        upper = beta_dist.ppf(1.0 - gamma / 2.0, n_eff * z + 1, n_eff * (1 - z))
    else:
        z = z_sim
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
