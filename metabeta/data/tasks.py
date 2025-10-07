import torch
from torch import distributions as D
import random
from metabeta.data.distributions import (
    Normal,
    StudentT,
    Uniform,
    Bernoulli,
    NegativeBinomial,
    ScaledBeta,
)
from metabeta.utils import fullCovary

# -----------------------------------------------------------------------------
probs = torch.tensor([0.10, 0.40, 0.05, 0.25, 0.10, 0.10])
dists = [
    Normal,
    StudentT,
    Uniform,
    Bernoulli,
    NegativeBinomial,
    ScaledBeta,
]


class Task:
    """base class for generating regression datasets"""

    def __init__(
        self,
        nu_ffx: torch.Tensor,  # ffx prior means
        tau_ffx: torch.Tensor,  # ffx prior stds
        tau_eps: torch.Tensor,  # noise prior std
        n_ffx: int,  # with bias
        features: torch.Tensor | None = None,
        limit: float = 300,  # try to sd(y) below this
        use_default: bool = False,  # use default prior parameters for predictors
        correlate: bool = True,  # correlate numerical predictors
    ):
        self.d = n_ffx
        self.features = features
        self.original_limit = limit
        self.limit = limit
        self.use_default = use_default
        self.correlate = correlate

        # data distribution
        idx = D.Categorical(probs).sample((n_ffx,))
        self.dist_data_ = [dists[i] for i in idx]
        self.dist_data = []
        self.cidx = []

        # ffx distribution
        assert len(nu_ffx) == len(tau_ffx) == self.d, "dimension mismatch"
        self.nu_ffx = nu_ffx
        self.tau_ffx = tau_ffx
        self.dist_ffx = D.Normal(self.nu_ffx, self.tau_ffx)

        # noise distribution
        self.tau_eps = tau_eps
        self.sigma_eps = D.HalfNormal(self.tau_eps).sample((1,))[0] + 1e-3  # type: ignore

    def sampleFfx(self) -> torch.Tensor:
        return self.dist_ffx.sample()

    def sampleFeatures(self, n_samples: int, ffx: torch.Tensor) -> torch.Tensor:
        """instantiate feature distributions within variance bounds and sample from them"""
        features = [torch.empty((n_samples, 0))]
        for i in range(len(ffx) - 1):
            weight = ffx[i + 1]
            dist = self.dist_data_[i](
                weight, limit=self.limit, use_default=self.use_default
            )
            if str(dist)[:4] == "Bern":
                self.cidx.append(i)
            self.dist_data += [dist]
            x = dist.sample((n_samples, 1))
            self.limit -= (x * weight).abs().max()
            features += [x]
        out = torch.cat(features, dim=-1)
        return out

    def induceCorrelation(self, x: torch.Tensor) -> torch.Tensor:
        """sample a correlation matrix and apply it to the continuous design matrix columns"""
        if self.d < 3 or not self.correlate:
            return x
        mean, std = x.mean(dim=0), x.std(dim=0)
        x_ = (x - mean) / std
        L = D.LKJCholesky(self.d - 1, 10.0).sample()

        # correlate continuous
        x_cor = (x_ @ L.T) * std + mean
        x_cor[:, self.cidx] = x[:, self.cidx]  # preserve categorial

        # correlate categorial
        R = L @ L.T
        for i in self.cidx:
            nidx = list(range(self.d - 1))
            nidx.pop(i)
            j = random.choice(nidx)
            x_cor[:, i] = self.correlateBinary(x_cor[:, j], R[i, j])
        return x_cor

    def correlateBinary(self, v: torch.Tensor, r: float | torch.Tensor):
        """generate a categorial variable whose correlation with variable v is r"""
        v = (v - v.mean()) / v.std()
        z = torch.randn_like(v)
        z = r * v + (1 - r**2) ** 0.5 * z
        probs = torch.sigmoid(z)
        z = torch.bernoulli(probs)
        return z

    def addIntercept(self, x: torch.Tensor):
        n_samples = x.shape[0]
        intercept = torch.ones(n_samples, 1)
        out = torch.cat([intercept, x], dim=-1)
        return out

    def sampleError(self, n_samples: int) -> torch.Tensor:
        eps = torch.randn((n_samples,))
        eps = (eps - torch.mean(eps)) / torch.std(eps)
        return eps * self.sigma_eps

    def signalToNoiseRatio(self, y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        eps = y - eta
        ess = (eta - eta.mean()).square().sum()
        rss = eps.square().sum()
        snr = ess / rss
        return snr

    def relativeNoiseVariance(self, y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        eps = y - eta
        return eps.var() / y.var()


# -----------------------------------------------------------------------------
