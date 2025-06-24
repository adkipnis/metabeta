from math import sqrt
import torch
from torch import distributions as D
from typing import Dict, Tuple, Union, List

# -----------------------------------------------------------------------------
# base class
class Task:
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor], # standard deviation of the additive noise
                 n_ffx: int, # without bias
                 ):

        # data distribution
        self.d = n_ffx
        self.dist_data = D.Uniform(-sqrt(12), sqrt(12)) # mean = 0, sd = 1

        # ffx distribution
        self.sigma_ffx = 3.
        self.dist_ffx = D.Normal(0., self.sigma_ffx)

        # noise distribution
        self.sigma_error = sigma_error
 
    def _sampleFfx(self) -> torch.Tensor:
        d = self.d + 1
        return self.dist_ffx.sample((d,)) # type: ignore

    def _sampleFeatures(self, n_samples: int) -> torch.Tensor:
        return self.dist_data.sample((n_samples, self.d)) # type: ignore

    def _addIntercept(self, x: torch.Tensor):
        n = x.shape[0]
        intercept = torch.ones(n, 1)
        return torch.cat([intercept, x], dim=-1)

    def _sampleError(self, n_samples: int) -> torch.Tensor:
        eps = torch.randn((n_samples,))
        eps = eps / torch.std(eps)
        return self.sigma_error * eps

    def sample(self, n_samples: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def signalToNoiseRatio(self, y: torch.Tensor, eta: torch.Tensor):
        eps = y - eta
        ess = torch.sum((eta - torch.mean(eta)).square())
        rss = torch.sum(eps.square())
        snr = ess / rss
        return snr


