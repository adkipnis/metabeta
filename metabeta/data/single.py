# generate a single hierarchical dataset
from dataclasses import dataclass
from random import choice
import torch
from torch import distributions as D
from metabeta.utils import fullCovary
from metabeta.data.distributions import (
    Normal,
    StudentT,
    LogNormal,
    Uniform,
    ScaledBeta,
    Bernoulli,
    NegativeBinomial,
)
from metabeta import plot


# -----------------------------------------------------------------------------
def standardize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x = x - x.mean(dim, keepdim=True)
    x = x / x.std(dim, keepdim=True)
    return x

def standardnormal(*size):
    x = torch.randn(size)
    return standardize(x)

def checkContinuous(x: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    diffs = (x - x.round()).abs()
    return (diffs > tol).all(dim=0)

# -----------------------------------------------------------------------------
@dataclass
class Prior:
    nu_ffx: torch.Tensor
    tau_ffx: torch.Tensor
    tau_eps: torch.Tensor
    tau_rfx: torch.Tensor

    def __post_init__(self):
        self.d = len(self.tau_ffx)
        self.q = len(self.tau_rfx)
        self.sigmas_rfx = None
        self.sigma_eps = None

    def sampleFfx(self) -> torch.Tensor:
        ffx = D.Normal(self.nu_ffx, self.tau_ffx).sample()
        return ffx

    def sampleRfx(self, m: int) -> torch.Tensor:
        self.sigmas_rfx = D.HalfCauchy(self.tau_rfx).sample()
        rfx = standardnormal(m, self.q)
        # rfx = torch.where(rfx.isnan(), 0, rfx)
        rfx *= self.sigmas_rfx.unsqueeze(0) # type: ignore
        return rfx

    def sampleEps(self, n: int) -> torch.Tensor:
        self.sigma_eps = D.HalfCauchy(self.tau_eps).sample()
        eps = standardnormal(n)
        eps *= self.sigma_eps # type: ignore
        return eps


# -----------------------------------------------------------------------------
dist_dict = {
    Normal: 0.05,
    StudentT: 0.25,
    LogNormal: 0.05,
    Uniform: 0.05,
    ScaledBeta: 0.15,
    Bernoulli: 0.20,
    NegativeBinomial: 0.15,
}

probs = torch.tensor(list(dist_dict.values()))
dists = list(dist_dict.keys())

@dataclass
class Design:
    toy: bool = False
    use_default: bool = False
    correlate: bool = True
 
    def __post_init__(self):
        self.dists = list()

    def column(self, n: int, parameter: float) -> torch.Tensor:
        idx = D.Categorical(probs).sample()
        dist = dists[idx](parameter, use_default=self.use_default)
        self.dists.append(dist)
        x = dist.sample((n,))
        return x

    def induceCorrelation(self, x: torch.Tensor) -> torch.Tensor:
        """sample a correlation matrix and apply it to the continuous design matrix columns"""
        d = x.shape[1]
        if d < 2:
            return x
        mean, std = x.mean(dim=0), x.std(dim=0)
        x_ = (x - mean) / std
        L = D.LKJCholesky(d, 10.0).sample()
        R = L @ L.T
 
        # correlate continuous
        x_cor = (x_ @ L.T) * std + mean
 
        # correlate categorical
        continuous = checkContinuous(x)
        idx_cont = torch.where(continuous)[0]
        idx_cat = torch.where(~continuous)[0]
        for i in idx_cat:
            if torch.rand(1) > 0.5 and len(idx_cont):
                j = choice(idx_cont)
                x_cor[:, i] = self.correlateBinary(x_cor[:, j], R[i, j])
            else:
                x_cor[:, i] = x[:, i]
        return x_cor
    
        x = torch.zeros(n, d)
        x[:, 0] = 1
        x[:, 1:] = torch.randn(n, d-1)
        return x


# -----------------------------------------------------------------------------
@dataclass
class Generator:
    prior: Prior
    design: Design
    n_i: list[int] | torch.Tensor # number of observations per group

    def __post_init__(self):
        self.d = self.prior.d # number of ffx
        self.q = self.prior.q # number of rfx
        self.m = len(self.n_i) # number of groups
        self.n_i = torch.tensor(self.n_i) # number of observations per group
        self.n = int(self.n_i.sum()) # total number of observations
 

    def sample(self) -> dict[str, torch.Tensor]:
        # globals
        ffx = self.prior.sampleFfx() # (d,)
        X = self.design.sample(self.n, self.d) # (n, d)
        eps = self.prior.sampleEps(self.n) # (n,)

        # locals
        rfx = self.prior.sampleRfx(self.m) # (m, q)
        groups = torch.repeat_interleave(torch.arange(self.m), self.n_i)  # type: ignore
        rfx_ext = rfx[groups]  # (n, q)
        Z = X[:, : self.q]

        # outcomes
        y_hat = X @ ffx + (Z * rfx_ext).sum(dim=-1) # (n,)
        y = y_hat + eps
        rnv = eps.var() / y.var()

        # Cov(mean Z, rfx), needed for standardization
        weighted_rfx = Z.mean(0, keepdim=True) * rfx
        cov = fullCovary(weighted_rfx)
        cov_sum = cov.sum() - cov[0, 0]

        # outputs
        out = {
            # data
            "X": X,  # (n, d-1)
            "y": y,  # (n,)
            "groups": groups,  # (n,)
            # params
            "ffx": ffx,  # (d,)
            "rfx": rfx,  # (m, q)
            "sigmas_rfx": rfx.std(0),  # (q,)
            "sigma_eps": eps.std(0),  # (1,)
            # priors
            "nu_ffx": self.prior.nu_ffx,  # (d,)
            "tau_ffx": self.prior.tau_ffx,  # (d,)
            "tau_rfx": self.prior.tau_rfx,  # (q,)
            "tau_eps": self.prior.tau_eps,  # (1,)
            # misc
            "m": torch.tensor(self.m),  # (1,)
            "n": torch.tensor(self.n),  # (1,)
            "n_i": self.n_i,  # (m,)
            "d": torch.tensor(self.d),  # (1,)
            "q": torch.tensor(self.q),  # (1,)
            "cov_sum": cov_sum,  # (1,)
            "rnv": rnv,  # (1,)
            "okay": torch.tensor(True),
        }
        return out


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # standardnormal
    x = standardnormal(10, 2)

    # prior params
    d = 3
    q = 2
    nu_ffx = torch.randn(d)
    tau_ffx = torch.randn(d).abs()
    tau_eps = torch.randn(1).abs()
    tau_rfx = torch.randn(q).abs()

    # prior
    m = 4
    N = 100
    prior = Prior(nu_ffx, tau_ffx, tau_eps, tau_rfx)
    beta = prior.sampleFfx()
    alpha = prior.sampleRfx(m)
    noise = prior.sampleEps(N)

    # design matrix
    design = Design()

    # generator
    n_i = [N//m for _ in range(m)]
    gen = Generator(prior, design, n_i)
    ds = gen.sample()


