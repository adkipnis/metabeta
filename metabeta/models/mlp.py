from typing import Callable, Sequence
import torch
from torch import nn


# --- initializers
Initializer = Callable[[nn.Module], None]
METHODS = ['kaiming', 'xavier', 'lecun']
DISTRIBUTIONS = ['uniform', 'normal']

def lecun_normal(weight: torch.Tensor) -> torch.Tensor:
    fan_in = nn.init._calculate_correct_fan(weight, 'fan_in')
    std = (1.0 / fan_in) ** 0.5
    return nn.init.normal_(weight, 0.0, std)

def lecun_uniform(weight: torch.Tensor) -> torch.Tensor:
    fan_in = nn.init._calculate_correct_fan(weight, 'fan_in')
    limit = (3.0 / fan_in) ** 0.5
    return nn.init.uniform_(weight, -limit, limit)

idict = {m: dict() for m in METHODS}
idict['kaiming']['uniform'] = nn.init.kaiming_uniform_
idict['kaiming']['normal'] = nn.init.kaiming_normal_
idict['xavier']['uniform'] = nn.init.xavier_uniform_
idict['xavier']['normal'] = nn.init.xavier_normal_
idict['lecun']['uniform'] = lecun_uniform
idict['lecun']['normal'] = lecun_normal

def getInitializer(method: str, distribution: str) -> Initializer:
    assert method in METHODS, 'unknown method'
    assert distribution in DISTRIBUTIONS, 'unknown distribution'
    init_func = idict[method][distribution]

    def initializer(layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            init_func(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    return initializer

def zeroInitializer(layer: nn.Module, limit: float = 1e-3) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, -limit, limit)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


# --- activations
