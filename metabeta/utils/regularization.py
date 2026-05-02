from typing import Literal
import math
import torch
from torch.nn import functional as F

THRESHOLD = 20.0
EPS = 1e-12

# exp/log
def maskedExp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, torch.exp(x), 0)


def maskedLog(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, torch.log(x), 0)


def logDetExp(x: torch.Tensor) -> torch.Tensor:
    """log |d exp(x)/dx| = x, masked for padded dims."""
    return torch.where(x != 0, x, torch.zeros_like(x))


# softplus
def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.softplus(x, beta=beta, threshold=THRESHOLD)


def inverseSoftplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    bx = beta * x
    return torch.where(
        bx > THRESHOLD,
        x,
        torch.expm1(bx).clamp_min(EPS).log() / beta,
    )


def maskedSoftplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    mask = x.ne(0).to(x.dtype)
    return softplus(x, beta=beta) * mask


def maskedInverseSoftplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    mask = x.ne(0).to(x.device)
    return inverseSoftplus(x, beta=beta) * mask


def logDetSoftplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """log |d softplus(x;beta)/dx| = log sigmoid(beta*x), masked for padded dims."""
    return torch.where(x != 0, F.logsigmoid(beta * x), torch.zeros_like(x))


# constrainer


def maskedSqrtSoftplus(x: torch.Tensor) -> torch.Tensor:
    mask = x.ne(0).to(x.dtype)
    return softplus(x).sqrt() * mask


def maskedInverseSqrtSoftplus(x: torch.Tensor) -> torch.Tensor:
    mask = x.ne(0).to(x.device)
    return inverseSoftplus(x.square()) * mask


def logDetSqrtSoftplus(x: torch.Tensor) -> torch.Tensor:
    """log |d sqrt(softplus(x))/dx|, masked for padded dims."""
    sp = softplus(x).clamp_min(EPS)
    val = F.logsigmoid(x) - 0.5 * sp.log() - math.log(2)
    return torch.where(x != 0, val, torch.zeros_like(x))


def getConstrainers(
    method: Literal['exp', 'softplus', 'softplus-sqrt'] = 'softplus',
):
    if method == 'exp':  # unbounded gradient; best when sigmas span orders of magnitude
        return maskedExp, maskedLog, logDetExp
    elif method == 'softplus':  # safe default; poor gradient for very small sigmas
        return maskedSoftplus, maskedInverseSoftplus, logDetSoftplus
    elif method == 'softplus-sqrt':  # compressed upper range; good for sigmas in [0.05, 3]
        return maskedSqrtSoftplus, maskedInverseSqrtSoftplus, logDetSqrtSoftplus
    raise ValueError(f'unknown constrainer method: {method}')


# correlation parameterization (LKJCholesky partial-correlation encoding)


def corrToUnconstrained(corr: torch.Tensor) -> torch.Tensor:
    """Correlation matrix (..., q, q) → unconstrained vector (..., q*(q-1)//2).

    Identity correlation maps to zero. Inverse: unconstrainedToCholeskyCorr.
    """
    q = corr.shape[-1]
    if q == 1:
        return corr.new_zeros(*corr.shape[:-2], 0)
    eye = torch.eye(q, dtype=corr.dtype, device=corr.device)
    L = torch.linalg.cholesky(corr + 1e-6 * eye)
    z_parts = []
    for i in range(1, q):
        row = L[..., i, :i]
        cumsum_sq = F.pad(row.pow(2)[..., :-1].cumsum(-1), (1, 0))
        denom = (1.0 - cumsum_sq).clamp(min=1e-8).sqrt()
        z_parts.append(torch.atanh((row / denom).clamp(-1 + 1e-6, 1 - 1e-6)))
    return torch.cat(z_parts, dim=-1)


def corrToLower(corr: torch.Tensor) -> torch.Tensor:
    """Correlation matrix (..., q, q) → lower-triangle values (..., q*(q-1)//2)."""
    q = corr.shape[-1]
    return torch.stack([corr[..., i, j] for i in range(q) for j in range(i)], dim=-1)


def unconstrainedToCholeskyCorr(z: torch.Tensor, q: int) -> torch.Tensor:
    """Unconstrained vector (..., q*(q-1)//2) → lower-triangular Cholesky (..., q, q).

    Correlation matrix is L @ L.mT. Inverse of corrToUnconstrained.
    """
    batch = z.shape[:-1]
    L = z.new_zeros(*batch, q, q)
    L[..., 0, 0] = 1.0
    cursor = 0
    for i in range(1, q):
        w = torch.tanh(z[..., cursor : cursor + i])
        cursor += i
        remaining = torch.ones(*batch, dtype=z.dtype, device=z.device)
        for j in range(i):
            L[..., i, j] = w[..., j] * remaining.clamp(min=1e-8).sqrt()
            remaining = remaining - L[..., i, j].pow(2)
        L[..., i, i] = remaining.clamp(min=1e-8).sqrt()
    return L


def corrLowerToFull(r: torch.Tensor, q: int) -> torch.Tensor:
    """Lower-triangle values (..., d_corr) → symmetric (..., q, q) correlation matrix."""
    batch = r.shape[:-1]
    corr = torch.eye(q, device=r.device, dtype=r.dtype).reshape((1,) * len(batch) + (q, q)).expand(*batch, q, q).clone()
    cursor = 0
    for i in range(1, q):
        for j in range(i):
            corr[..., i, j] = r[..., cursor]
            corr[..., j, i] = r[..., cursor]
            cursor += 1
    return corr


def corrLowerToUnconstrained(r: torch.Tensor, q: int) -> torch.Tensor:
    """Lower-triangle correlation values (..., d_corr) → unconstrained (..., d_corr)."""
    return corrToUnconstrained(corrLowerToFull(r, q))


def logDetJacobianCorr(z: torch.Tensor, q: int) -> torch.Tensor:
    """log |d corrToLower(L L^T) / dz| for the LKJ partial-correlation encoding.

    For q == 2: r = tanh(z), so log|dr/dz| = log(1 - tanh^2(z)) summed to a scalar.
    Returns shape (...) — one scalar per batch element.
    """
    if q <= 1:
        return z.new_zeros(z.shape[:-1])
    if q == 2:
        return torch.log1p(-torch.tanh(z).pow(2).clamp(max=1 - 1e-7)).sum(-1)
    raise NotImplementedError(f'logDetJacobianCorr not implemented for q={q}')


# crunching
def dampen(x: torch.Tensor, p: float = 0.45) -> torch.Tensor:
    return x.sign() * x.abs().pow(p)


def squish(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    WIDE_BETA = 0.5

    x = torch.randn(64, 8)

    # softplus (beta=1)
    y = softplus(x)
    z = inverseSoftplus(y)
    assert torch.allclose(x, z, atol=1e-6), 'softplus not invertible'

    # softplus (beta=0.5)
    y = softplus(x, beta=WIDE_BETA)
    z = inverseSoftplus(y, beta=WIDE_BETA)
    assert torch.allclose(x, z, atol=1e-5), 'wide softplus not invertible'

    # masked softplus
    mask = torch.randint(0, 2, size=(64, 8)).bool()
    x[mask] = 0.0
    y = maskedSoftplus(x)
    z = maskedInverseSoftplus(y)
    assert torch.allclose(x, z, atol=1e-6), 'masked softplus not invertible'

    # masked wide softplus
    y = maskedSoftplus(x, beta=WIDE_BETA)
    z = maskedInverseSoftplus(y, beta=WIDE_BETA)
    assert torch.allclose(x, z, atol=1e-5), 'masked wide softplus not invertible'

    # masked sqrt softplus
    y = maskedSqrtSoftplus(x)
    z = maskedInverseSqrtSoftplus(y)
    assert torch.allclose(x, z, atol=1e-5), 'masked sqrt softplus not invertible'

    # masked log
    y = maskedExp(x)
    z = maskedLog(y)
    assert torch.allclose(x, z, atol=1e-6), 'masked log not invertible'

    x = torch.arange(-100, 100, step=200 / 512)
    y = dampen(x)
    plt.plot(x, y, label='dampen')
    assert torch.isfinite(y).all(), 'dampen is not finite'

    y = squish(x)
    plt.plot(x, y, label='squish')
    plt.legend()
    assert torch.isfinite(y).all(), 'squish is not finite'
