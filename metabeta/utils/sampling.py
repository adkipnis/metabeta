import numpy as np
from scipy.stats import wishart


def sampleCounts(n: int, m: int, alpha: float = 10.) -> np.ndarray:
    ''' draw ns (= counts) for m groups, such that the sum is n '''
    p = np.random.dirichlet(np.ones(m) * alpha)
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
        return sampleCounts(n, m, alpha)
    return ns


def counts2groups(ns: np.ndarray) -> np.ndarray:
    ''' convert array of counts to group index array '''
    m = len(ns)
    unique = np.arange(m)
    groups = np.repeat(unique, ns)
    return groups


def logUniform(a: float,
               b: float,
               add: float = 0.0,
               round: bool = False) -> float|int:
    ''' truncated log-uniform with optional rounding '''
    assert a > 0, 'lower bound must be positive'
    assert b > a, 'upper bound must be larger than lower bound'
    out = np.exp(np.random.uniform(np.log(a), np.log(b)))
    out += add
    if round:
        return int(np.round(out))
    return float(out)


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

