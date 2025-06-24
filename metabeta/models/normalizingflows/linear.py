import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class Transform(nn.Module):
    def forward(self,
                x: torch.Tensor,
                condition: torch.Tensor|None = None,
                mask: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError

    def inverse(self,
                z: torch.Tensor,
                condition: torch.Tensor|None = None,
                mask: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None]:
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
        z = s * x + t
        log_det = s.abs().log()
        if mask is not None:
            log_det = log_det * mask
        return z, log_det.sum(-1), mask

    def inverse(self, z, condition=None, mask=None):
        s, t = self.scale.expand_as(z), self.bias.expand_as(z)
        # s, t = self.scale, self.bias
        if mask is not None:
            t = t * mask
        x = (z - t) / s
        log_det = -s.abs().log()
        if mask is not None:
            log_det = log_det * mask
        return x, log_det.sum(-1), mask


class Permute(Transform):
    # adapted from https://github.com/bayesflow/bayesflow to handle masking
    def __init__(self, target_dim: int, mode: str = 'shuffle'):
        super().__init__()
        self.target_dim = target_dim
        assert mode in ['shuffle', 'swap'], "unkown mode selected"
        self.mode = mode
        self.pivot = target_dim//2
        self.inv_pivot = target_dim - self.pivot
        if self.mode == 'shuffle':
            self.perm = torch.randperm(target_dim)
            self.inv_perm = self.perm.argsort()
 
    def forward(self, x, condition=None, mask=None):
        if self.mode == 'shuffle':
            z = x[..., self.perm]
            if mask is not None:
                mask = mask[..., self.perm]
        elif self.mode == 'swap':
            z1, z2 = x[..., :self.pivot], x[..., self.pivot:]
            z = torch.cat([z2, z1], dim=-1)
            if mask is not None: 
                mask1, mask2 = mask[..., :self.pivot], mask[..., self.pivot:]
                mask = torch.cat([mask2, mask1], dim=-1)
        else:
            raise ValueError
        log_det = torch.zeros(z.shape[:-1], device=z.device)
        return z, log_det, mask

    def forwardMask(self, mask):
        if self.mode == 'shuffle':
            mask = mask[..., self.perm]
        elif self.mode == 'swap':
            mask1, mask2 = mask[..., :self.pivot], mask[..., self.pivot:]
            mask = torch.cat([mask2, mask1], dim=-1)
        return mask

    def inverse(self, z, condition=None, mask=None):
        if self.mode == 'shuffle':
            x = z[..., self.inv_perm]
            if mask is not None:
                mask = mask[..., self.inv_perm]
        elif self.mode == 'swap':
            x1, x2 = z[..., :self.inv_pivot], z[..., self.inv_pivot:]
            x = torch.cat([x2, x1], dim=-1)
            if mask is not None: 
                mask1, mask2 = mask[..., :self.inv_pivot], mask[..., self.inv_pivot:]
                mask = torch.cat([mask2, mask1], dim=-1)
        else:
            raise ValueError
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        return x, log_det, mask


