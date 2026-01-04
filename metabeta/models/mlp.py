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


class MLP(nn.Module):
    ''' Multi-Layer Perceptron with optional skip-connection:
        [SLP] * len(d_hidden) -> Linear -> Dropout
    '''
    def __init__(
        self,
        d_input: int,
        d_hidden: int | Sequence[int],
        d_output: int,
        use_bias: bool = True,
        layer_norm: bool = False,
        pre_norm: bool = False,
        activation: str = 'ReLU',
        dropout: float = 0.0,
        residual: bool = False,
        residual_scale: float = 0.1,
        shortcut: bool = False, # additional linear layer from input to output
        weight_init: tuple[str, str] | None = None,
        zero_init: bool = False,
    ):
        super().__init__()
        if isinstance(d_hidden, int):
            d_hidden = [d_hidden]

        # build layers
        layers = []
        d_in = d_input
        for d_out in d_hidden:
            ff = Feedforward(
                d_input=d_in,
                d_output=d_out,
                use_bias=use_bias,
                layer_norm=layer_norm,
                pre_norm=pre_norm,
                activation=activation,
                dropout=dropout,
                residual=residual,
                residual_scale=residual_scale,
            )
            layers += [ff]
            d_in = d_out

        # add final layer
        layers += [nn.Linear(d_in, d_output, bias=use_bias),
                   nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)

        # optionally add skip_connection
        self.shortcut = None
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Linear(d_input, d_output, bias=use_bias),
                nn.Dropout(dropout),
            )

        # optionally initialize weights
        if weight_init is not None:
            initializer = getInitializer(*weight_init)
            self.apply(initializer)

        # optionally zero-init final layer
        if zero_init:
            zeroInitializer(self.layers[-2])
            if self.shortcut is not None:
                zeroInitializer(self.shortcut[0])

    def forward(self, x):
        h = self.layers(x)
        if self.shortcut is not None:
            h = h + self.shortcut(x)
        return h


# --- special cases

class TransformerFFN(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        use_bias: bool = True,
        activation: str = 'GELU',
        shortcut: bool = True,
    ):
        super().__init__()
        assert activation in ['GELU', 'GeGLU'], 'invalid activation'
        self.net = MLP(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            use_bias=use_bias,
            activation=activation,
            shortcut=shortcut,
            # defaults:
            layer_norm=True,
            pre_norm=True,
            dropout=0.01,
            residual=True,
            residual_scale=0.1,
            weight_init=('xavier', 'normal'),
            zero_init=False,
        )

    def forward(self, x):
        return self.net(x)


class FlowMLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int | Sequence[int],
        d_output: int,
        use_bias: bool = True,
        shortcut: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            use_bias=use_bias,
            shortcut=shortcut,
            # defaults:
            layer_norm=False,
            pre_norm=False,
            activation='ELU',
            dropout=0.0,
            residual=True,
            residual_scale=0.1,
            weight_init=('lecun', 'normal'),
            zero_init=True,
        )

    def forward(self, x, context: torch.Tensor | None = None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.net(x)


# --------------------------------------------------------
if __name__ == '__main__':

    def run(cls):
        model = cls(**cfg)
        out = model(x)
        model = torch.compile(model)
        model(x)
        return out

    # initializer
    intializer = getInitializer('xavier', 'normal')
    intializer = getInitializer('lecun', 'uniform')
    intializer = getInitializer('lecun', 'normal')

    # dims
    b = 64
    d_input = 32
    d_hidden = [16, 8]
    d_output = 4
    x = torch.randn(b, d_input)

    # Base Feedforward Layer
    cfg = {'d_input': d_input,
           'd_output': d_output,
           'activation': 'ReLU'}
    run(Feedforward)

    cfg.update({'layer_norm': True, 'activation': 'GELU'})
    run(Feedforward)

    cfg.update({'pre_norm': True, 'residual': False})
    run(Feedforward)

    # Base MLP
    cfg.update({'d_hidden': d_hidden, 'weight_init': None})
    run(MLP)

    cfg.update({'d_hidden': d_hidden[0],
                'shortcut': True,
                'weight_init': ('xavier', 'normal'),
                'zero_init': True})
    run(MLP)

    # TransformerFFN
    cfg = {'d_input': d_input, 'd_hidden': d_hidden[0], 'd_output': d_output}
    run(TransformerFFN)

    cfg.update({'activation': 'GeGLU'})
    run(TransformerFFN)

    # FlowMLP
    cfg = {'d_input': d_input, 'd_hidden': d_hidden, 'd_output': d_output}
    run(FlowMLP)

    x, context = x.chunk(2, dim=-1)
    model = FlowMLP(**cfg)
    torch.compile(model)
    y = model(x, context)

