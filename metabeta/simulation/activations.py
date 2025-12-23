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
