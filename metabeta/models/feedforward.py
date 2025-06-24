from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F


def initializer(method: str, distribution: str):
    assert method in ['kaiming', 'xavier'] and distribution in ['uniform', 'normal'], 'invalid weight initializer args'
    if method == 'kaiming':
        if distribution == 'uniform':
            init = nn.init.kaiming_uniform_
        elif distribution == 'normal':
            init = nn.init.kaiming_normal_
    elif method == 'xavier':
        if distribution == 'uniform':
            init = nn.init.xavier_uniform_
        elif distribution == 'normal':
            init = nn.init.xavier_normal_
    def initialize(layer):
        if isinstance(layer, nn.Linear):
            init(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return initialize


class SLP(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 activation: str = 'Mish',
                 dropout: float = 0.01,
                 norm: str | None = None,
                 skip: bool = False,
                 **kwargs
                 ):
        super().__init__()
        layer = [nn.Linear(d_input, d_output)]
        if norm == 'batch':
            layer += [nn.BatchNorm1d(d_output, eps=1e-3)]
        elif norm == 'layer':
            layer += [nn.LayerNorm(d_output, eps=1e-3)]
        layer += [eval(f'nn.{activation}()')]
        layer += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layer)
        # optional skip connection
        self.skip = skip
        if self.skip:
            self.shortcut = nn.Linear(d_input, d_output)

    def forward(self, x):
        h = self.net(x)
        if self.skip:
            h += self.shortcut(x)
        return h


