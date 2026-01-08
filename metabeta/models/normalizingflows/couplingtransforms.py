from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.feedforward import FlowMLP, FlowResidualNet


class CouplingTransform(nn.Module):
    ''' Base class for coupling transforms:
        - build conditioner network
        - propose transform parameters
        - directionally apply transform parameters to inputs
    '''
    split_dims: tuple[int, int]
    d_context: int = 0
    net_kwargs: dict = {}

    @abstractmethod
    def _build(self, net_kwargs: dict) -> None:
        ...

    @abstractmethod
    def _propose(
        self, x1: torch.Tensor, condition: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask2: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...

    def inverse(self, x1, x2, condition=None, mask2=None):
        return self(x1, x2, condition, mask2, inverse=True)


class Affine(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
        alpha: float = 1.0, # softclamping denominator
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.alpha = alpha

        # setup conditioner
        self._build(dict(net_kwargs))

    def _build(self, net_kwargs: dict):
        net_type = net_kwargs['net_type']
        assert net_type in ['mlp', 'residual']
        net_kwargs['d_output'] = 2 * self.split_dims[1]

        # MLP Conditioner
        if net_type == 'mlp':
            net_kwargs.update({
                'd_input': self.split_dims[0] + self.d_context,
                'd_hidden': (net_kwargs['d_ff'],) * net_kwargs['depth'],
            })
            self.conditioner = FlowMLP(**net_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            net_kwargs.update({
                'd_input': self.split_dims[0],
                'd_context': self.d_context,
                'd_hidden': net_kwargs['d_ff'],
            })
            self.conditioner = FlowResidualNet(**net_kwargs)

    def _propose(self, x1, condition=None):
        parameters = self.conditioner(x1, condition)
        log_s, t = parameters.chunk(2, dim=-1)
        log_s = self.alpha * torch.tanh(log_s / self.alpha)  # softclamping
        return dict(log_s=log_s, t=t)

    def forward(self, x1, x2, condition=None, mask2=None, inverse=False):
        params = self._propose(x1, condition)
        log_s, t = params['log_s'], params['t']
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        if inverse:
            x2 = (x2 - t) * (-log_s).exp()
        else:
            x2 = log_s.exp() * x2 + t
        log_det = log_s.sum(-1)
        return x2, log_det


class RationalQuadratic(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
        n_bins: int = 16,
        tail_bound: float = 3.0,
        min_val: float = 1e-3,
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.min_val = min_val # bin width, height, derivative
        self.min_total = n_bins * min_val # width, height
        self.d_ff = net_kwargs['d_ff']
        self.tail_constant = np.log(np.exp(1 - self.min_val) - 1)
        assert self.min_total < 1.0, 'either lower min_val or number of bins'

        # setup conditioner
        self._build(dict(net_kwargs))

    def _build(self, net_kwargs: dict):
        net_type = net_kwargs['net_type']
        assert net_type in ['mlp', 'residual']
        net_kwargs['d_output'] = (3 * self.n_bins - 1) * self.split_dims[1]

        # MLP Conditioner
        if net_type == 'mlp':
            net_kwargs.update({
                'd_input': self.split_dims[0] + self.d_context,
                'd_hidden': (net_kwargs['d_ff'],) * net_kwargs['depth'],
            })
            self.conditioner = FlowMLP(**net_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            net_kwargs.update({
                'd_input': self.split_dims[0],
                'd_context': self.d_context,
                'd_hidden': net_kwargs['d_ff'],
            })
            self.conditioner = FlowResidualNet(**net_kwargs)

