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
