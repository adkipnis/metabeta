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
        alpha = rng.uniform(2.0, 20.0)  # vary entropy across datasets
    p = rng.dirichlet(np.ones(m) * alpha)
    ns = np.round(p * n).astype(int)
    diff = n - ns.sum()
    if diff > 0:
        idx = ns.argmin(0)
        ns[idx] += diff
    elif diff < 0:
        idx = ns.argmax(0)
        ns[idx] += diff
    if (ns < 1).any():  # try again
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


def skewedBeta(
    rng: np.random.Generator,
    low: float,
    high: float,
    mode: float,
    concentration: float,
    size: int | tuple[int, ...],
) -> np.ndarray:
    """Sample from Beta distribution rescaled to [low, high] with given mode.

    concentration=1 gives uniform on [low, high] regardless of mode.
    concentration>1 concentrates mass around the mode.
    """
    assert low < mode < high, 'mode must be strictly between low and high'
    assert concentration >= 1, 'concentration must be >= 1'
    t = (mode - low) / (high - low)
    a = 1 + (concentration - 1) * t
    b = 1 + (concentration - 1) * (1 - t)
    return low + (high - low) * rng.beta(a, b, size=size)


def spikeAndSlab(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    p_zero: float = 0.90,
    scale: float = 1.0,
) -> np.ndarray:
    """Spike-and-slab: p_zero mass on 0, rest from Laplace(0, scale)"""
    spike = rng.random(size=size) < p_zero
    slab = rng.laplace(loc=0, scale=scale, size=size)
    return np.where(spike, 0.0, slab)


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
    """Sample a correlation matrix from LKJ(eta) using torch LKJCholesky."""
    if d < 1:
        raise ValueError('d must be >= 1')
    if eta <= 0:
        raise ValueError('eta must be > 0')

    if d == 1:
        return np.ones((1, 1))

    # Seed torch from numpy RNG so LKJ sampling is reproducible under numpy seed control.
    seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
    concentration = torch.tensor(float(eta), dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        chol = torch.distributions.LKJCholesky(d, concentration=concentration).sample()

    corr = (chol @ chol.T).cpu().numpy()
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return corr


def samplePermutation(
    rng: np.random.Generator,
    d: int,
) -> np.ndarray:
    perm = rng.permutation(d - 1) + 1
    zero = np.zeros((1,), dtype=int)
    perm = np.concatenate([zero, perm])
    return perm
