import torch
import numpy as np
from scipy.stats import binom

from metabeta.utils.evaluation import Proposal


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
