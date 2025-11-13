# generate a single hierarchical dataset
from dataclasses import dataclass
import torch
from torch import distributions as D
from metabeta.utils import fullCovary


# -----------------------------------------------------------------------------
def standardize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x = x - x.mean(dim, keepdim=True)
    x = x / x.std(dim, keepdim=True)
    return x

def standardnormal(*size):
    x = torch.randn(size)
    return standardize(x)

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
        sigmas_rfx = D.HalfNormal(self.tau_rfx).sample()
        rfx = standardnormal(m, self.q)
        # rfx = torch.where(rfx.isnan(), 0, rfx)
        rfx *= sigmas_rfx.unsqueeze(0)
        return rfx
    
    def sampleEps(self, n: int) -> torch.Tensor:
        sigma_eps = D.HalfNormal(self.tau_eps).sample()
        eps = standardnormal(n)
        eps *= sigma_eps
        return eps


# -----------------------------------------------------------------------------
@dataclass
class Design:
    correlate: bool = False
    
    def sample(self, n: int, d: int) -> torch.Tensor:
        x = torch.zeros(n, d)
        x[:, 0] = 1
        x[:, 1:] = torch.randn(n, d-1)
        return x
 
        
# -----------------------------------------------------------------------------
@dataclass
class Generator:
    prior: Prior
    design: Design
    n_i: list[int] # number of observations per group
    
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
        groups = torch.repeat_interleave(torch.arange(self.m), self.n_i)  # (n,)
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
