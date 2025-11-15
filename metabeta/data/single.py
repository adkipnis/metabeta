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

def checkBinary(x: torch.Tensor) -> torch.Tensor:
    is_binary = (x == 0) | (x == 1)
    return is_binary.all(dim=0)

def correlateBinary(v: torch.Tensor, r: float | torch.Tensor):
    """generate a categorical variable whose correlation with variable v is r"""
    v = (v - v.mean()) / v.std()
    z = torch.randn_like(v)
    z = r * v + (1 - r**2) ** 0.5 * z
    probs = torch.sigmoid(z)
    z = torch.bernoulli(probs)
    return z

def counts2groups(n_i: torch.Tensor) -> torch.Tensor:
    unique = torch.arange(len(n_i))
    groups = torch.repeat_interleave(unique, n_i)
    return groups

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
class Synthesizer:
    ''' sample a design matrix and groups using synthetic distributions '''
    toy: bool = False
    use_default: bool = False
    correlate: bool = True

    def column(self, n: int, parameter: float) -> torch.Tensor:
        idx = D.Categorical(probs).sample()
        dist = dists[idx](parameter, use_default=self.use_default)
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
                x_cor[:, i] = correlateBinary(x_cor[:, j], R[i, j])
            else:
                x_cor[:, i] = x[:, i]
        return x_cor
 
    def sampleX(self, n: int, d: int, parameters: torch.Tensor) -> torch.Tensor:
        ''' sample design matrix from synthetic distributions '''
        # init design matrix
        x = torch.zeros(n, d)
        x[:, 0] = 1.

        # fill remaining columns
        if self.toy:
            x[:, 1:] = torch.randn(n, d-1)
            return x
        else:
            for i in range(1, d):
                x[:, i] = self.column(n, parameters[i].item())

        # correlate columns
        if self.correlate:
            x[:, 1:] = self.induceCorrelation(x[:, 1:])
        return x

    def sample(self, d: int, n_i: torch.Tensor, parameters: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.sampleX(n=int(n_i.sum()), d=d, parameters=parameters)
        groups = counts2groups(n_i)
        return x, groups, n_i



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
        X = self.design.sample(self.n, self.d, ffx) # (n, d)

        # locals
        rfx = self.prior.sampleRfx(self.m) # (m, q)
        groups = torch.repeat_interleave(torch.arange(self.m), self.n_i)  # type: ignore
        rfx_ext = rfx[groups]  # (n, q)
        Z = X[:, : self.q]

        # outcome
        y_hat = X @ ffx + (Z * rfx_ext).sum(dim=-1) # (n,)
        signal = y_hat.var()

        # calibrate noise
        okay = torch.tensor(False)
        attempts = 0
        while not okay:
            eps = self.prior.sampleEps(self.n) # (n,)
            noise = self.prior.sigma_eps.square()
            rnv = noise / (signal + noise)
            attempts += 1
            okay = (0.1 <= rnv <= 0.9) or torch.tensor(attempts > 25)
        y = y_hat + eps

        # Cov(mean Z, rfx), needed for standardization
        weighted_rfx = Z.mean(0, keepdim=True) * rfx
        cov = fullCovary(weighted_rfx)
        cov_sum = cov.sum() - cov[0, 0]

        # visualize
        # plot.dataset(
        #     torch.cat([y.unsqueeze(-1), X[:, 1:]], dim=-1),
        #     names=['y'] + [f'x{j}' for j in range(1, self.d)]
        # )

        # outputs
        out = {
            # data
            "X": X,  # (n, d-1)
            "y": y,  # (n,)
            "groups": groups,  # (n,)
            # params
            "ffx": ffx,  # (d,)
            "rfx": rfx,  # (m, q)
            "sigmas_rfx": self.prior.sigmas_rfx,  # (q,)
            "sigma_eps": self.prior.sigma_eps,  # (1,)
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
            "categorical": checkBinary(X[:, 1:]), # (d-1,)
            "rnv": rnv,  # (1,)
            "okay": okay,
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


