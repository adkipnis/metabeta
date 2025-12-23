''' simplified version of SCM activations from https://github.com/soda-inria/tabicl'''
import numpy as np
import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F
from metabeta.utils import logUniform

# --- preprocessing layers
class Standardizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-6
        return (x - self.mean) / self.std

class RandomScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = (2 * torch.randn(1)).exp()
        self.bias = torch.randn(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x + self.bias)

class RandomScaleFactory:
    def __init__(self, act: nn.Module):
        self.act = act

    def __call__(self) -> nn.Module:
        return nn.Sequential(Standardizer(), RandomScale(), self.act())


# --- random choice layers
class RandomChoice(nn.Module):
    ''' pass data through one of {n_choice} activations '''
    def __init__(self, acts: list[nn.Module], n_choice: int = 1):
        super().__init__()
        assert len(acts), 'provided empty list of activations'
        assert n_choice > 0, 'number of choices must be positive'
        self.acts = acts
        self.n = len(acts)
        self.k = min(n_choice, self.n)
        
    def __repr__(self) -> str:
        return f'Random-{self.k}'

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        u = torch.arange(0, n) % self.k
        mask = F.one_hot(u, num_classes=self.k).float()
        perm = torch.randperm(n)
        return mask[perm]
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.randint(0, self.n, (self.k,))
        acts = [self.acts[idx]() for idx in indices]
        out = torch.stack([act(x) for act in acts], dim=-1)
        mask = self.mask(x)
        return (out * mask).sum(-1)

class RandomChoiceFactory:
    def __init__(self, acts: list[nn.Module], n_choice: int = 1):
        self.acts = acts
        self.n_choice = n_choice

    def __call__(self) -> nn.Module:
        return RandomChoice(self.acts, self.n_choice)


# --- simple activations
class Abs(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs()

class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()

class SqrtAbs(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().sqrt()

class Exp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.exp()

class LogAbs(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().log()

class SE(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (-x.square()).exp()

class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin()
    
class Cos(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.cos()

class Mod(nn.Module):
    def __init__(self, lower: float = 0., upper: float = 10.):
        super().__init__()
        self.k = np.random.uniform(lower, upper)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x % self.k

class Sign(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, 1., -1.)

class Step(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.ceil()

class UnitInterval(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x.abs() <= 1, 1., 0.)

# --- probabilistic activations
class Bernoulli(nn.Module):
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

simple_activations = [
