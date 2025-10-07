import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class Transform(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def inverse(
        self,
        z: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def forwardMask(self, mask: torch.Tensor):
        return mask


class ActNorm(Transform):
    # adapted from https://github.com/bayesflow/bayesflow to handle masking
    def __init__(self, target_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((target_dim,)))
        self.bias = nn.Parameter(torch.zeros((target_dim,)))

    def forward(self, x, condition=None, mask=None):
        s, t = self.scale.expand_as(x), self.bias.expand_as(x)
        if mask is not None:
            t = t * mask
        x = s * x + t
        log_det = s.abs().log()
        if mask is not None:
            log_det = log_det * mask
        return x, log_det.sum(-1), mask

    def inverse(self, z, condition=None, mask=None):
        s, t = self.scale.expand_as(z), self.bias.expand_as(z)
        # s, t = self.scale, self.bias
        if mask is not None:
            t = t * mask
        z = (z - t) / s
        log_det = -s.abs().log()
        if mask is not None:
            log_det = log_det * mask
        return z, log_det.sum(-1), mask


class Permute(Transform):
