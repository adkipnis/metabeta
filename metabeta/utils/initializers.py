from typing import Callable
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


IDICT = {m: dict() for m in METHODS}
IDICT['kaiming']['uniform'] = nn.init.kaiming_uniform_
IDICT['kaiming']['normal'] = nn.init.kaiming_normal_
IDICT['xavier']['uniform'] = nn.init.xavier_uniform_
IDICT['xavier']['normal'] = nn.init.xavier_normal_
IDICT['lecun']['uniform'] = lecun_uniform
IDICT['lecun']['normal'] = lecun_normal


def getInitializer(method: str, distribution: str) -> Initializer:
    assert method in METHODS, 'unknown method'
    assert distribution in DISTRIBUTIONS, 'unknown distribution'
    init_func = IDICT[method][distribution]

    def initializer(layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            init_func(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    return initializer


def zeroInitializer(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def lastZeroInitializer(layers: nn.ModuleList) -> None:
    for m in reversed(layers):
        if isinstance(m, nn.Linear):
            zeroInitializer(m)
            break


def weightNormInitializer(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        if not torch.all(layer.weight == 0):
            nn.utils.parametrizations.weight_norm(layer)
