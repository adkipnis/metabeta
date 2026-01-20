import numpy as np
from scipy.stats import wishart


def sampleCounts(rng: np.random.Generator,
                 n: int, m: int, alpha: float = 10.) -> np.ndarray:
    ''' draw ns (= counts) for m groups, such that the sum is n '''
    p = rng.dirichlet(np.ones(m) * alpha)
    ns = np.round(p * n).astype(int)
    diff = n - ns.sum()
    if diff > 0:
        idx = ns.argmin(0)
        ns[idx] += diff
    elif diff < 0:
        idx = ns.argmax(0)
        ns[idx] += diff
    if (ns < 1).any(): # try again
        print('non-positive counts found')
        return sampleCounts(rng, n, m, alpha)
    return ns


def counts2groups(ns: np.ndarray) -> np.ndarray:
    ''' convert array of counts to group index array '''
    m = len(ns)
    unique = np.arange(m)
    groups = np.repeat(unique, ns)
    return groups


def truncLogUni(rng: np.random.Generator,
                low: float,
                high: float,
                size: int | tuple[int, ...],
                add: float = 0.0,
                round: bool = False,
                ) -> np.ndarray:
    ''' sample from log uniform in [low, high) + {add} and optionally floor to integer '''
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
    ''' sample a correlation matrix from the Wishart distribution '''
    if nu is None:
        nu = d + 1  # minimal df for SPD

    # sample covariance matrix
    S = wishart(df=nu, scale=np.eye(d)).rvs(random_state=rng)

    # convert to correlation matrix
    std = np.sqrt(np.diag(S))
    C = S / np.outer(std, std)

    # ensure numerical safety
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.)
    return C

