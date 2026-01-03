import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import numpy as np
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
        x = torch.einsum('...nd,dap->...nap', x, self.w)
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
                 levels: int = 3
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
                 levels: int = 3
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


# --- stochastic post-hoc layers
class Stochastic(Base):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 standardize: bool = False,
                 sigma: float = 0.01,
                 ):
        super().__init__(n_in, n_out, standardize)
        self.sigma = sigma # noise standard deviation

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = super().preprocess(x)
        x = x + torch.randn_like(x) * self.sigma
        return x

class Categorical(Stochastic):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]

        # include reference category and get probs
        zeros = torch.zeros_like(x[..., 0:1])
        x = torch.cat([x, zeros], dim=-1)
        probs = F.softmax(x, dim=-1)

        # dummy code categories
        x = probs.argmax(-1)
        x = F.one_hot(x, num_classes=self.n_out + 1)[..., 1:]
        return x


class Poisson(Stochastic):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        lam = x.exp()
        x = torch.poisson(lam)
        return x


class Geometric(Stochastic):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)[..., 0]
        x = D.Geometric(logits=x).sample()
        return x

class NegativeBinomial(Stochastic):
    @property
    def n_param(self) -> int:
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, r = self.preprocess(x).split(1, dim=-1)
        p = p.squeeze(-1)
        r = F.softplus(r.squeeze(-1))
        x = D.NegativeBinomial(total_count=r, logits=p).sample()
        return x


def getPosthocLayers() -> list[nn.Module]:
    deterministic = [Threshold, MultiThreshold, QuantileBins, Rank]
    stochastic = [Categorical, Poisson, Geometric, NegativeBinomial]
    return deterministic + stochastic


if __name__ == '__main__':
    from metabeta.utils import setSeed
    import numpy as np
    from scipy.stats import spearmanr

    setSeed(1)

    def test(model, x, statistic='pearson'):
        y = torch.cat([model(x), model(x)], dim=-1)
        R = 0
        for i in range(len(y)):
            if statistic == 'pearson':
                corr = np.corrcoef(y[i], rowvar=False)
            elif statistic == 'spearman':
                corr = spearmanr(y[i], axis=0)[0]
            else:
                raise ValueError('unknown statistic')
            R += corr
        R /= len(y)
        unique = R[np.triu_indices_from(R, k=1)]
        print(unique)

    # init
    batch, n, d = 64, 100, 8
    x = torch.randn((batch, n, d)) * 3

    # --- deterministic tests
    model = Threshold(d, 2)
    y = model(x)

    model = MultiThreshold(d, 1, levels=3)
    y = model(x)

    model = QuantileBins(d, 2, levels=2)
    y = model(x)

    model = Rank(d, 5)
    y = model(x)

    # --- stochastic tests
    model = Categorical(d, 1)
    test(model, x)

    model = Categorical(d, 2)
    test(model, x)

    model = Poisson(d, 1)
    test(model, x)

    model = Geometric(d, 1)
    test(model, x)

    model = NegativeBinomial(d, 1)
    test(model, x)

