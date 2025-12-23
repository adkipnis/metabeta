import numpy as np
import torch
from torch import nn
from typing import Iterable


class CauseSampler(nn.Module):
    def __init__(self,
                 n_samples: int,
                 n_causes: int,
                 dist: str = 'normal', # [mixed, normal, uniform]
                 fixed: bool = False, # random parameters for dist
                 **kwargs
                 ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_causes = n_causes
        self.dist = {'normal': self._normal,
                     'uniform': self._uniform,
                     'mixed': self._mixed}[dist]
        self.fixed = fixed
        if not self.fixed:
            self.mu = torch.randn(n_causes)
            self.sigma = (torch.randn(n_causes) * self.mu).abs()

    def _normal(self, shape: Iterable) -> torch.Tensor:
        x = torch.randn(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def _uniform(self, shape: Iterable) -> torch.Tensor:
        x = torch.rand(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + (x-0.5) * sigma * np.sqrt(12)
        return x

    def _multinomial(self, shape: Iterable) -> torch.Tensor:
        n, d = shape
        n_categories = np.random.randint(2, 20)
        probs = torch.rand((d, n_categories))
        x = torch.multinomial(probs, n, replacement=True).permute(1,0).float()
        x = (x - x.mean(0)) / x.std(0)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def _zipf(self, shape: Iterable) -> torch.Tensor:
        a = 2 * np.random.rand() + 2
        x = np.random.zipf(a, shape) # type: ignore
        x = torch.tensor(x).clamp(max=10).float()
        x = (x - x.mean(0)) / x.std(0)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

