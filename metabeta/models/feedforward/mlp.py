from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F

from metabeta.utils.activations import getActivation
from metabeta.utils.initializers import (
    getInitializer, zeroInitializer, weightNormInitializer)


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
        rscale: float = 0.1, # λ
    ):
        super().__init__()
        self.residual = residual if d_input == d_output else False
        if self.residual:
            self._rscale = nn.Parameter(torch.tensor(rscale))
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
        if dropout:
            layers += [nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)
 
    @property
    def rscale(self) -> torch.Tensor:
        return F.softplus(self._rscale).clamp(max=1)

    def forward(self, x):
        if self.residual:
            return x + self.rscale * self.layers(x)
        return self.layers(x)


class MLP(nn.Module):
    ''' Multi-Layer Perceptron with optional skip-connection:
        [FF] * len(d_hidden) -> Linear -> Dropout
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
        shortcut: bool = False, # additional linear layer from input to output
        weight_init: tuple[str, str] | None = None,
        zero_init: bool = False,
        weight_norm: bool = False,
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
        if zero_init: # this should only be used for residual setups
            zeroInitializer(self.layers[-2])
            if self.shortcut is not None:
                zeroInitializer(self.shortcut[0])

        # optionally apply weight norm to linear layers
        if weight_norm:
            self.apply(weightNormInitializer)


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
        dropout: float = 0.01,
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
            dropout=dropout,
            shortcut=shortcut,
            # defaults:
            layer_norm=True,
            pre_norm=True,
            residual=False,
            weight_init=('xavier', 'normal'),
            weight_norm=False,
            zero_init=True,
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
        activation: str = 'ELU',
        dropout: float = 0.0,
        shortcut: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            use_bias=use_bias,
            activation=activation,
            dropout=dropout,
            shortcut=shortcut,
            zero_init=zero_init,
            # defaults:
            layer_norm=False,
            pre_norm=False,
            residual=True,
            weight_init=('lecun', 'normal'),
            weight_norm=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
        if (mask is not None) and (context is not None):
            x = torch.cat([x, mask, context], dim=-1)
        elif mask is not None:
            x = torch.cat([x, mask], dim=-1)
        elif context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.net(x)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # dims
    d_input = 32
    d_hidden = [16, 8]
    d_output = 4
    x = torch.randn(64, d_input)

    # TransformerFFN
    cfg = {'d_input': d_input, 'd_hidden': d_hidden[0], 'd_output': d_output}
    model = TransformerFFN(**cfg) # type: ignore
    out = model(x)

    # FlowMLP
    cfg = {'d_input': d_input, 'd_hidden': d_hidden, 'd_output': d_output}
    x, context = x.chunk(2, dim=-1)
    model = FlowMLP(**cfg)
    out = model(x, context)

