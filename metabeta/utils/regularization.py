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
