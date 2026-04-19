import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


class BaseDist(nn.Module):
    """Base distribution for normalizing flows: Normal or StudentT, with optional trainable parameters.

    When d_context > 0, a zero-initialized linear head maps the context to per-sample
    additive corrections to the global loc and log_scale parameters.  This lets the
    flow model only the residual around the context-predicted mean/scale rather than
    the full conditional distribution.
    """

    def __init__(
        self,
        d_data: int,
        family: str = 'normal',
        trainable: bool = True,
        d_context: int = 0,
    ) -> None:
        super().__init__()
        assert family in ('normal', 'student'), f'unknown family: {family}'
        self.family = family
        self.d_data = d_data
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
        # optional context-conditional correction: zero-init so it starts as global base
        self.context_head: nn.Linear | None = None
        if d_context > 0:
            self.context_head = nn.Linear(d_context, 2 * d_data)
            nn.init.zeros_(self.context_head.weight)
            nn.init.zeros_(self.context_head.bias)

    def __repr__(self) -> str:
        kind = 'Trainable' if self._loc.requires_grad else 'Static'
        ctx = f', d_context={self.context_head.in_features}' if self.context_head else ''
        with torch.no_grad():
            scale = (F.softplus(self._log_scale) + 1e-6).mean().item()
            s = f'loc={self._loc.mean().item():.2f}, scale={scale:.2f}'
            if self.family == 'student':
                df = (F.softplus(self._log_df) + 3.0).mean().item()
                s = f'df={df:.2f}, {s}'
        return f'{kind}{self.family.title()}({s}{ctx})'

    def _params(
        self, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (loc, scale) with optional context-conditional correction."""
        loc = self._loc
        log_scale = self._log_scale
        if self.context_head is not None and context is not None:
            delta = self.context_head(context)
            loc = loc + delta[..., : self.d_data]
            log_scale = log_scale + delta[..., self.d_data :]
        return loc, F.softplus(log_scale) + 1e-6

    def _dist(self, context: torch.Tensor | None = None) -> D.Distribution:
        loc, scale = self._params(context)
        if self.family == 'student':
            return D.StudentT(F.softplus(self._log_df) + 3.0, loc, scale)
        return D.Normal(loc, scale)

    def sample(self, shape: tuple[int, ...], context: torch.Tensor | None = None) -> torch.Tensor:
        # shape = (*batch_dims, d_data)
        # Sample a zero-mean unit variate and apply loc/scale so loc/scale broadcast correctly
        # regardless of whether they have context-induced batch dims.
        with torch.no_grad():
            loc, scale = self._params(context)
            if self.family == 'student':
                df = F.softplus(self._log_df) + 3.0
                z = D.StudentT(df).sample(shape[:-1])  # (*batch_dims, d_data), df broadcasts
            else:
                z = torch.randn(shape, device=loc.device, dtype=loc.dtype)
            return loc + scale * z

    def logProb(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self._dist(context).log_prob(x)


# --------------------------------------------------------
if __name__ == '__main__':
    b, d = 8, 3
    dist = BaseDist(d, family='student', trainable=True)
    x = dist.sample((b, d))
    log_prob = dist.logProb(x)
    print(dist, x.shape, log_prob.shape)
