import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


class BaseDist(nn.Module):
    """Base distribution for normalizing flows: Normal or StudentT, with optional trainable parameters."""

    def __init__(self, d_data: int, family: str = 'normal', trainable: bool = True) -> None:
        super().__init__()
        assert family in ('normal', 'student'), f'unknown family: {family}'
        self.family = family
        self._loc = nn.Parameter(torch.zeros(d_data), requires_grad=trainable)
        # softplus^{-1}(1) ≈ 0.541; initialises scale ≈ 1
        self._log_scale = nn.Parameter(
            torch.full((d_data,), torch.log(torch.expm1(torch.tensor(1.0))).item()),
            requires_grad=trainable,
        )
        if family == 'student':
            # softplus^{-1}(6) ≈ 5.999; initialises df = softplus(5.999) + 3 ≈ 9
            self._log_df = nn.Parameter(
                torch.full((d_data,), torch.log(torch.expm1(torch.tensor(6.0))).item()),
                requires_grad=trainable,
            )

    def __repr__(self) -> str:
        kind = 'Trainable' if self._loc.requires_grad else 'Static'
        with torch.no_grad():
            scale = (F.softplus(self._log_scale) + 1e-6).mean().item()
            s = f'loc={self._loc.mean().item():.2f}, scale={scale:.2f}'
            if self.family == 'student':
                df = (F.softplus(self._log_df) + 3.0).mean().item()
                s = f'df={df:.2f}, {s}'
        return f'{kind}{self.family.title()}({s})'

    def _dist(self) -> D.Distribution:
        scale = F.softplus(self._log_scale) + 1e-6
        if self.family == 'student':
            return D.StudentT(F.softplus(self._log_df) + 3.0, self._loc, scale)
        return D.Normal(self._loc, scale)

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        # shape = (*batch_dims, d_data); sample (*batch_dims,) from the d_data-vector distribution
        with torch.no_grad():
            return self._dist().sample(shape[:-1])

    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist().log_prob(x)


# --------------------------------------------------------
if __name__ == '__main__':
    b, d = 8, 3
    dist = BaseDist(d, family='student', trainable=True)
    x = dist.sample((b, d))
    log_prob = dist.logProb(x)
    print(dist, x.shape, log_prob.shape)
