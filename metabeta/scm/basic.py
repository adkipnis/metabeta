import numpy as np
import torch
from torch import nn

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

