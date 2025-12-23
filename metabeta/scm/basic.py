import numpy as np
import torch
from torch import nn
from torch import distributions as D

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

basic_activations = [
    nn.Identity,
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.SELU,
    nn.SiLU,
    nn.ELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardtanh,
    nn.Softplus,
    Abs,
    Square,
    SqrtAbs,
    Exp,
    LogAbs,
    SE,
    Sine,
    Cos,
    Mod,
    Sign,
    Step,
    UnitInterval,
]

stochastic_activations = [
    Bernoulli,
    Poisson,
    Geometric,
    Gamma,
    Beta
]
