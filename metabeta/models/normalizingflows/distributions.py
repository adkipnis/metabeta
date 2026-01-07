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

