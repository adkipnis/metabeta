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


