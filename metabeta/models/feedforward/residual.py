import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.utils import (
    getActivation, getInitializer,
    zeroInitializer, lastZeroInitializer, weightNormInitializer)

class ResidualBlock(nn.Module):
    ''' Residual Block with optional GLU:
        Linear -> Norm -> Activation -> Dropout -> Linear -> Norm -> (GLU ->) Residual Sum
    '''
    def __init__(
        self,
        d_hidden: int,
        d_context: int = 0,
        use_bias: bool = True,
        layer_norm: bool = False,
        pre_norm: bool = False,
        eps: float = 1e-3, # numerical stability in layer norm denominator
        activation: str = 'ReLU',
        dropout: float = 0.0,
        use_glu: bool = False,
        rscale: float = 0.1, # residual scale
        cscale: float = 0.1, # context scale
    ):
        super().__init__()
        self._rscale = nn.Parameter(torch.tensor(rscale))
        self.cscale = cscale

        # main layers
        layers = []
        if layer_norm and pre_norm:
            layers += [nn.LayerNorm(d_hidden, eps=eps, elementwise_affine=True)]
        layers += [nn.Linear(d_hidden, d_hidden, bias=use_bias)]
        if layer_norm and not pre_norm:
            layers += [nn.LayerNorm(d_hidden, eps=eps, elementwise_affine=True)]
        layers += [getActivation(activation)]
        if dropout:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(d_hidden, d_hidden, bias=use_bias)]
        self.layers = nn.Sequential(*layers)

        # projection layer for GLU
        self.proj = None
        if d_context and use_glu:
            self.proj = nn.Linear(d_context, d_hidden, bias=use_bias)

    @property
    def rscale(self) -> torch.Tensor:
        return F.softplus(self._rscale).clamp(max=1)
    
    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.layers(x)
        if context is not None and self.proj is not None:
            ctx = self.cscale * self.proj(context)
            h = torch.cat([h, ctx], dim=-1)
            h = F.glu(h, dim=-1)
        h = x + self.rscale * h
        return h


class ResidualNet(nn.Module):
    ''' Residual Feedforward Network:
        Linear -> Dropout -> [RB] * n_blocks -> Linear -> Dropout
    '''
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        n_blocks: int,
        d_context: int = 0,
        use_bias: bool = True,
        layer_norm: bool = False,
        pre_norm: bool = False,
        activation: str = 'ReLU',
        dropout: float = 0.01,
        use_glu: bool = False,
        weight_init: tuple[str, str] | None = None,
        zero_init: bool = False,
        weight_norm: bool = False,
    ):
        super().__init__()

        # input and output projection
        self.proj_in = nn.Sequential(
            nn.Linear(d_input + d_context, d_hidden, bias=use_bias),
            nn.Dropout(dropout))
        self.proj_out = nn.Sequential(
            nn.Linear(d_hidden, d_output, bias=use_bias),
            nn.Dropout(dropout))

        # main blocks
        blocks = [
            ResidualBlock(
                d_hidden=d_hidden,
                d_context=d_context,
                use_bias=use_bias,
                layer_norm=layer_norm,
                pre_norm=pre_norm,
                activation=activation,
                dropout=dropout,
                use_glu=use_glu,
            )
            for _ in range(n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

        # optionally initialize weights
        if weight_init is not None:
            initializer = getInitializer(*weight_init)
            self.apply(initializer)

        # optionally zero-init final layer
        if zero_init:
            zeroInitializer(self.proj_out[0])
            for block in self.blocks:
                lastZeroInitializer(block.layers) # type: ignore

        # optionally apply weight norm to linear layers
        if weight_norm:
            self.apply(weightNormInitializer)


    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, context)
        h = self.proj_out(h)
        return h


class FlowResidualNet(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        n_blocks: int,
        d_context: int = 0,
        use_bias: bool = True,
        use_glu: bool = True,
    ):
        super().__init__()
        self.net = ResidualNet(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            n_blocks=n_blocks,
            d_context=d_context,
            use_bias=use_bias,
            use_glu=use_glu,
            # defaults:
            layer_norm=True,
            pre_norm=True,
            activation='ELU',
            dropout=0.0,
            weight_init=('lecun', 'normal'),
            weight_norm=True,
            zero_init=True,
        )

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x, context)


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def run(cls):
        model = cls(**cfg)
        out = model(x, context)
        model = torch.compile(model)
        return out

    # dims
    b = 64
    d_input = 16
    d_hidden = 32
    d_output = 8
    n_blocks = 3
    x = torch.randn(b, d_hidden)
    context = None

    # --- Residual Block
    # basic
    cfg = {'d_hidden': d_hidden}
    run(ResidualBlock)

    # with context, dropout and layer norm
    d_context = 5
    context = torch.randn(b, d_context)
    cfg.update({'d_context': d_context, 'layer_norm': True, 'dropout': 0.01}) # type: ignore
    run(ResidualBlock)

    # with GLU
    cfg.update({'use_glu': True})
    run(ResidualBlock)

    # --- Residual Net
    cfg.update({
        'd_input': d_input,
        'd_output': d_output,
        'n_blocks': n_blocks,
    })
    x = torch.randn(b, d_input)
    run(ResidualNet)

    # --- Flow Residual Net
    cfg = {
        'd_input': d_input,
        'd_hidden': d_hidden,
        'd_output': d_output,
        'n_blocks': n_blocks,
        'd_context': d_context,
    }
    run(FlowResidualNet)


