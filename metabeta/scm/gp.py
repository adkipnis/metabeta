import numpy as np
import torch
from torch import nn
from torch import distributions as D
from metabeta.utils import logUniform
from .meta import Standardizer


# --- GP based activations
class MaternKernel:
    def __init__(self) -> None:
        self.df = np.random.choice([1,3,5]) / 2

    def __repr__(self) -> str:
        return f'Matern-{self.df}'

    def __call__(self, k: int, ell: float):
        scale = self.df ** 0.5 / ell
        freqs = D.StudentT(df=self.df).sample((k,)) * scale
        factor = (2 / k) ** 0.5
        return freqs, factor

class SEKernel: # squared exponential / RBF
    def __repr__(self) -> str:
        return 'SE'

    def __call__(self, k: int, ell: float):
        freqs = torch.randn(k) / ell
        factor = (2 / k) ** 0.5
        return freqs, factor

class FractKernel: # scale-free fractional kernel
    def __repr__(self) -> str:
        return 'Fractional'

    def __call__(self, k: int, ell: float):
        freqs = k * torch.rand(k)
        decay_exponent = -logUniform(0.7, 3.0)
        factor = freqs ** decay_exponent
        factor = factor / (factor ** 2).sum().sqrt()
        return freqs, factor


class GP(nn.Module):
    # sample from a GP with a random kernel [SE, Matern, Fractal]
    def __init__(self, k: int = 512, gp_type: str = ''):
        super().__init__()
        self.standardizer = Standardizer()
        self.kernels = {
            'Matern': MaternKernel,
            'SE': SEKernel,
            'Fract': FractKernel,
        }

        # choose kernel
        if gp_type:
            assert gp_type in self.kernels, f'Kernel not found in {self.kernels.keys()}'
        else:
            gp_type = np.random.choice(
                list(self.kernels.keys()),
                p=[0.5, 0.2, 0.3]
                # p=[0.0, 1.0, 0.0]
            )
        self.kernel = self.kernels[gp_type]()

        # setup parameters
        ell = logUniform(0.1, 16.0)
        self.freqs, factor = self.kernel(k, ell)
        self.bias = 2 * torch.pi * torch.rand(k)
        self.weight = factor * torch.randn(k)

    def __repr__(self) -> str:
        return f'GP-{self.kernel}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardizer(x)
        phi = torch.cos(self.freqs * x.unsqueeze(-1) + self.bias)
        return phi @ self.weight



