import logging
import random
import numpy as np
from scipy.stats import wishart
import torch

logger = logging.getLogger(__name__)


def setSeed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sampleCounts(
    rng: np.random.Generator, n: int, m: int, alpha: float | None = None
) -> np.ndarray:
    """draw ns (= counts) for m groups, such that the sum is n"""
    if alpha is None:
        alpha = rng.uniform(2.0, 20.0)   # vary entropy across datasets
    p = rng.dirichlet(np.ones(m) * alpha)
    ns = np.round(p * n).astype(int)
    diff = n - ns.sum()
    if diff > 0:
        idx = ns.argmin(0)
        ns[idx] += diff
    elif diff < 0:
        idx = ns.argmax(0)
        ns[idx] += diff
    if (ns < 1).any():   # try again
        logger.info('non-positive counts found')
        return sampleCounts(rng, n, m, alpha)
    return ns


def counts2groups(ns: np.ndarray) -> np.ndarray:
    """convert array of counts to group index array"""
    m = len(ns)
    unique = np.arange(m)
    groups = np.repeat(unique, ns)
    return groups


def truncLogUni(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...],
    add: float = 0.0,
    round: bool = False,
) -> np.ndarray:
    """sample from log uniform in [low, high) + {add} and optionally floor to integer"""
    assert 0 < low, 'lower bound must be positive'
    assert low <= high, 'lower bound smaller than upper bound'
    log_low = np.log(low)
    log_high = np.log(high)
    out = rng.uniform(log_low, log_high, size)
    out = np.exp(out) + add
    if round:
        out = np.floor(out).astype(int)
    return out


def wishartCorrelation(
    rng: np.random.Generator,
    d: int,
    nu: int | None = None,
) -> np.ndarray:
    """sample a correlation matrix from the Wishart distribution"""
    if nu is None:
        nu = d + 1  # minimal df for SPD

    # sample covariance matrix
    S = wishart(df=nu, scale=np.eye(d)).rvs(random_state=rng)

    # convert to correlation matrix
    std = np.sqrt(np.diag(S))
    C = S / np.outer(std, std)

    # ensure numerical safety
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    return C


def lkjCorrelation(
    rng: np.random.Generator,
    d: int,
    eta: float = 1.0,
) -> np.ndarray:
    """Sample a correlation matrix from LKJ(eta) using the vine method.

    See Lewandowski, Kurowicka, Joe (2009).
    eta=1 is uniform over correlation matrices, eta>1 shrinks toward identity.
    """
    if d == 1:
        return np.ones((1, 1))

    # sample partial correlations via Beta distribution
    L = np.zeros((d, d))
    for i in range(1, d):
        for j in range(i):
            alpha = eta + (d - 1 - i) / 2
            partial = 2 * rng.beta(alpha, alpha) - 1
            L[i, j] = partial

    # convert partial correlations to Cholesky factor
    C = np.zeros((d, d))
    C[0, 0] = 1.0
    for i in range(1, d):
        remaining = 1.0
        for j in range(i):
            C[i, j] = L[i, j] * np.sqrt(remaining)
            remaining *= 1 - L[i, j] ** 2
        C[i, i] = np.sqrt(remaining)

    # correlation matrix = C @ C.T
    corr = C @ C.T
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return corr


def samplePermutation(
    rng: np.random.Generator,
    d: int,
):
    perm = rng.permutation(d - 1) + 1
    zero = np.zeros((1,), dtype=int)
    perm = np.concatenate([zero, perm])
    return perm
