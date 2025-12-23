import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from metabeta.scm.meta import Standardizer


# --- deterministic post-hoc layers

class Base(nn.Module):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 standardize: bool = False):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.standardize = standardize
        if standardize:
            self.standardizer = Standardizer()
        alpha = torch.ones(n_in)
        self.w = D.Dirichlet(alpha).sample((n_out, self.n_param)).permute(2,0,1)

    @property
    def n_param(self) -> int:
        return 1

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            x = self.standardizer(x)
        x = torch.einsum('bnd,dap->bnap', x, self.w)
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(torch.sigmoid(x))

class Poisson(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.poisson(torch.where(x > 0, x**2, 0))

class Geometric(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(x) + 1e-9
        return D.Geometric(probs=p).sample().sqrt()

class Gamma(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return D.Gamma(2, x.exp()).sample().sqrt()

class Beta(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(x + 1e-9)
        alpha = m * 10
        beta = (1-m) * 10
        return D.Beta(alpha, beta).sample()


