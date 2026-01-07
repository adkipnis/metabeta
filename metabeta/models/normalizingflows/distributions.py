import numpy as np
import scipy
import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


class StaticDist(nn.Module):
    def __init__(self, d_data: int, family: str) -> None:
        super().__init__()
        assert family in ['normal', 'student'], 'distribution family unknown'
        self.d_data = d_data
        self.family = family
        self._sampling_dist = self._dist['scipy'](**self._default_params)
        self._training_dist = self._dist['torch'](**self._default_params)
        self.device = self._training_dist.mean.device

    def __repr__(self) -> str:
        params = ''
        for k,v in self._default_params.items():
            params += f'{k}: {v:.1f}, '
        return f'Static {self.family.title()} ({params[:-2]})'

    @property
    def _default_params(self) -> dict[str, float]:
        out = {}
        if self.family == 'student':
            out['df'] = 5.
        out['loc'] = 0.
        out['scale'] = 1.
        return out

    @property
    def _dist(self) -> dict:
        if self.family == 'normal':
            return {'torch': D.Normal, 'scipy': scipy.stats.norm} 
        elif self.family == 'student':
            return {'torch': D.StudentT, 'scipy': scipy.stats.t}
        else:
            raise ValueError

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
