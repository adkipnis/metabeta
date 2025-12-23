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


class Threshold(Base):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        x = (x > 0).float()
        return x


class MultiThreshold(Base):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 standardize: bool = False,
                 levels: int = 2
                 ):
        super().__init__(n_in, n_out, standardize)
        self.levels = levels # number of thresholds
        self.tau = np.sort(np.random.normal(size=levels-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        y = torch.zeros_like(x)
        for t in self.tau:
            y = y + (x > t)
        return y


class QuantileBins(Base):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 standardize: bool = False,
                 levels: int = 2
                 ):
        super().__init__(n_in, n_out, standardize)
        self.levels = levels # number of quantiles
        quantiles = np.sort(np.random.random(size=levels-1))
        self.quantiles = torch.tensor(quantiles).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        thresholds = torch.quantile(x.flatten(), self.quantiles)
        x = torch.bucketize(x, thresholds)
        return x


class Rank(Base):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = min(self.n_out, x.shape[-1])
        x = torch.topk(x, k).indices
        return x


class Gamma(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return D.Gamma(2, x.exp()).sample().sqrt()

class Beta(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(x + 1e-9)
        alpha = m * 10
        beta = (1-m) * 10
        return D.Beta(alpha, beta).sample()


