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
class GeGLU(nn.Module):
    ''' GELU-based Gated Linear Unit '''
    def forward(self, x): # 2d -> d
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)

ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish,
    'ELU': nn.ELU, # default for NF
    'GELU': nn.GELU, # default for Transformers
    'GeGLU': GeGLU, # optional for Set Transformers
}

def getActivation(name: str) -> nn.Module:
    assert name in ACTIVATIONS, f'Unknown activation {name}'
    return ACTIVATIONS[name]()


# --- multi-layer perceptron
class Feedforward(nn.Module):
    ''' Feedforward Layer:
        Linear -> Norm -> Activation -> Dropout (-> Residual)
    '''
    def __init__(
        self,
        d_input: int,
        d_output: int,
        use_bias: bool = True,
        layer_norm: bool = False,
        pre_norm: bool = False, # layer norm before linear layer
        eps: float = 1e-3, # numerical stability in layer norm denominator
        activation: str = 'ReLU',
        dropout: float = 0.0,
        residual: bool = False, # if true: x + λ * self.layers(x)
        residual_scale: float = 0.1, # λ
    ):
        super().__init__()
        self.residual = residual if d_input == d_output else False
        self.residual_scale = residual_scale
        if activation == 'GeGLU':
            d_output *= 2

        # construct
        layers = []
        if pre_norm and layer_norm:
            layers += [nn.LayerNorm(d_input, eps=eps, elementwise_affine=True)]
        layers += [nn.Linear(d_input, d_output, bias=use_bias)]
        if not pre_norm and layer_norm:
            layers += [nn.LayerNorm(d_output, eps=eps, elementwise_affine=True)]
        layers += [getActivation(activation)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            return x + self.residual_scale * self.layers(x)
        return self.layers(x)


