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


class MLP(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_hidden: int | Iterable,
                 d_output: int,
                 activation: str = 'Mish',
                 dropout: float = 0.01,
                 norm: str | None = None,
                 act_on_last: bool = False,
                 skip: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(d_hidden, int):
            d_hidden = [d_hidden]
        layers = []
        d_prev = d_input
        for d_next in d_hidden:
            layers += [SLP(d_prev, d_next, activation, dropout, norm, skip)]
            d_prev = d_next
        layers += [nn.Linear(d_prev, d_output)]
        if act_on_last:
            layers += [eval(f'nn.{activation}()')]
        self.net = nn.Sequential(*layers)
        # optional skip connection
        self.skip = skip
        if self.skip:
            self.shortcut = nn.Linear(d_input, d_output)
        self.apply(initializer('kaiming', 'uniform'))

    def forward(self, x: torch.Tensor, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = self.net(x)
        if self.skip:
            h += self.shortcut(x)
        return h


class ResidualBlock(nn.Module):
    def __init__(self,
                 d_data: int,
                 d_context: int = 0,
                 activation: str = 'Mish',
                 dropout: float = 0.01,
                 norm: str | None = None,
                 use_glu: bool = False,
                 ):
        super().__init__()
        self.use_glue = use_glu
        act = eval(f'nn.{activation}')
        layers = [nn.Linear(d_data, d_data)]
        if norm == 'batch':
            layers += [nn.BatchNorm1d(d_data, eps=1e-3)]
        elif norm == 'layer':
            layers += [nn.LayerNorm(d_data, eps=1e-3)]
        layers += [act()]
        layers += [nn.Linear(d_data, d_data)]
        if norm == 'batch':
            layers += [nn.BatchNorm1d(d_data, eps=1e-3)]
        elif norm == 'layer':
            layers += [nn.LayerNorm(d_data, eps=1e-3)]
        layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        if d_context > 0 and use_glu:
            self.proj = nn.Linear(d_context, d_data)
        self.act = act()

    def forward(self, x: torch.Tensor, context=None):
        h = self.net(x)
        if context is not None and self.use_glue:
            cat = torch.cat([h, self.proj(context)], dim=-1)
            h = F.glu(cat, dim=-1)
        h = self.act(h + x)
        return h


class ResidualNet(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_context: int = 0,
                 d_hidden: int = 64,
                 n_blocks: int = 2,
                 activation: str = 'Mish',
                 dropout: float = 0.01,
                 norm: str | None = None,
                 use_glu: bool = False,
                 zero_init: bool = True,
                 ):
        super().__init__()
        self.proj_in = nn.Linear(d_input + d_context, d_hidden)
        blocks = [
            ResidualBlock(d_data=d_hidden, d_context=d_context, use_glu=use_glu,
                          activation=activation, dropout=dropout, norm=norm, )
            for _ in range(n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.proj_out = nn.Linear(d_hidden, d_output)
        self.apply(initializer('kaiming', 'uniform'))
        if zero_init:
            nn.init.uniform_(self.proj_out.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.proj_out.bias, -1e-3, 1e-3)

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, context)
        h = self.proj_out(h)
        return h


