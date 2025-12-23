from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
from torch import nn
from metabeta.scm import CauseSampler


class NoiseLayer(nn.Module):
    def __init__(self, sigma: float | torch.Tensor):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        noise = torch.randn_like(x) * self.sigma
        return x + noise


@dataclass
class SCM(nn.Module):
    ''' simplified version of the MLP-based structural causal model
    in https://github.com/soda-inria/tabicl'''
    def __init__(self,
                 # data dims
                 n_samples: int,
                 n_features: int,

                 # causes
                 n_causes: int = 10, # number of units in initial layer
                 cause_dist: str = 'uniform', # [mixed, normal, uniform]
                 fixed: bool = False, # fixed moments of causes

                 # MLP architecture
                 n_layers: int = 8,
                 n_hidden: int = 32,
                 activation: Callable = nn.Tanh,

                 # weight initialization and feature extraction
                 sigma_w: float = 1.0, # for weight initialization
                 contiguous: bool = False, # sample adjacent features
                 blockwise: bool = True, # use blockwise dropout
                 p_dropout: float = 0.1, # dropout probability for weights

                 # Gaussian noise
                 sigma_e: float = 0.01, # for additive noise
                 vary_sigma_e: bool = True, # allow noise to vary per units
                 ):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_causes = n_causes
        self.cause_dist = cause_dist
        self.fixed = fixed
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = activation
        self.sigma_w = sigma_w
        self.contiguous = contiguous
        self.blockwise = blockwise
        self.p_dropout = p_dropout
        self.sigma_e = sigma_e
        self.vary_sigma_e = vary_sigma_e
 
        # make sure to have enough hidden units
        self.n_hidden = max(self.n_hidden, 2 * self.n_features)

        # init sampler for root nodes
        self.cs = CauseSampler(self.n_samples, self.n_causes,
                               dist=self.cause_dist,
                               fixed=self.fixed)

        # build layers
        layers = [self._buildLayer(self.n_causes)]
        for _ in range(self.n_layers - 1):
            layers += [self._buildLayer()]
        self.layers = nn.Sequential(*layers)

        # initialize weights
        with torch.no_grad():
            self._initLayers()


    def _buildLayer(self, input_dim: int = 0) -> nn.Module:
