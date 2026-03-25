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


def posteriorCorrelation(
    rfx: torch.Tensor,  # (b, m, s, q)
    mask_m: torch.Tensor,  # (b, m)
) -> torch.Tensor:
    """compute pairwise correlation matrices from posterior rfx samples"""
    mask = mask_m[:, :, None, None].float()   # (b, m, 1, 1)
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)   # (b, 1, 1, 1)
    mean = (rfx * mask).sum(dim=1, keepdim=True) / n   # (b, 1, s, q)
    centered = (rfx - mean) * mask   # (b, m, s, q)
    cov = torch.einsum('bmsi,bmsj->bsij', centered, centered) / n.view(-1, 1, 1, 1)   # (b, s, q, q)
    std = cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12).sqrt()   # (b, s, q)
    return cov / (std.unsqueeze(-1) * std.unsqueeze(-2))   # (b, s, q, q)


def evaluateCorrelation(
