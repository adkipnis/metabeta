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
