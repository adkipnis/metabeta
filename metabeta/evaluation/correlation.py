import torch
import numpy as np


def _batchCorrcoef(
    x: np.ndarray,  # (n_sim, m, q)
) -> np.ndarray:
    a, b = x[..., 0], x[..., 1]   # (n_sim, m)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a**2).sum(axis=-1) * (b**2).sum(axis=-1)).clip(min=1e-12)
    return num / den   # (n_sim)


def _nullCorrelations(m: int, n_sim: int = 2000, seed: int = 0) -> np.ndarray:
    """Empirical null distribution of |r| when rho=0, for a given m.
    Returns sorted (n_sim,) array of absolute correlation values."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sim, m, 2))
    null = np.abs(_batchCorrcoef(z))
    null.sort()
    return null


