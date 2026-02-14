import torch


# exp/log
def maskedExp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, torch.exp(x), 0)


def maskedLog(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, torch.log(x), 0)


# softplus
def softplus(x: torch.Tensor, limit: float = 20.0, eps: float = 1e-6) -> torch.Tensor:
    return torch.where(x > limit, x, torch.log(torch.exp(x + eps) + 1))


def inverseSoftplus(x: torch.Tensor, limit: float = 20.0, eps: float = 1e-6) -> torch.Tensor:
    return torch.where(x > limit, x, torch.log(torch.exp(x) - 1) - eps)


def maskedSoftplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, softplus(x), 0)


def maskedInverseSoftplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, inverseSoftplus(x), 0)


# crunching
def dampen(x: torch.Tensor, p: float = 0.45) -> torch.Tensor:
    return x.sign() * x.abs().pow(p)


def squish(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x = torch.randn(64, 8)

    # softplus
    y = softplus(x)
    z = inverseSoftplus(y)
    assert torch.allclose(x, z, atol=1e-6), 'softplus not invertible'

    # masked softplus
    mask = torch.randint(0, 2, size=(64, 8)).bool()
    x[mask] = 0.0
    y = maskedSoftplus(x)
    z = maskedInverseSoftplus(y)
    assert torch.allclose(x, z, atol=1e-6), 'masked softplus not invertible'

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
