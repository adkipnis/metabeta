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


