import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.normalizingflows import Transform


class ActNorm(Transform):
    def __init__(self, d_target: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((d_target,)))
        self.bias = nn.Parameter(torch.zeros((d_target,)))

    def __call__(self, x, condition=None, mask=None, inverse=False):
        scale, bias = self.scale.expand_as(x), self.bias.expand_as(x)
        if mask is not None:
            bias = bias * mask
        if inverse:
            x = (x - bias) / scale
        else:
            x = scale * x + bias
        log_det = scale.abs().log()
        if inverse:
            log_det = -log_det
        if mask is not None:
            log_det = log_det * mask
        return x, log_det.sum(-1), mask


class Permute(Transform):
    def __init__(self, d_target: int):
        super().__init__()
        self.d_target = d_target
        self.pivot = d_target // 2
        self.inv_pivot = d_target - self.pivot
        self.perm = torch.randperm(d_target)
        self.inv_perm = self.perm.argsort()

    def __call__(self, x, condition=None, mask=None, inverse=False):
        perm = self.inv_perm if inverse else self.perm
        x = x[..., perm]
        if mask is not None:
            mask = mask[..., perm]
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        return x, log_det, mask

    def forwardMask(self, mask):
        mask = mask[..., self.perm]
        return mask


