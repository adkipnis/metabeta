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


