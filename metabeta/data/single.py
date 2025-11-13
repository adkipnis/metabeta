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
 
        
