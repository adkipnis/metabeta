from abc import abstractmethod
import torch
from torch import nn
from metabeta.models.feedforward import FlowMLP, FlowResidualNet


class CouplingTransform(nn.Module):
    ''' Base class for coupling transforms:
        - build conditioner network
        - propose transform parameters
        - directionally apply transform parameters to inputs
    '''
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context

        # setup conditioner
        assert net_kwargs['net_type'] in ['mlp', 'residual']
        self._build(dict(net_kwargs))

    @abstractmethod
    def _build(self, net_kwargs: dict) -> None:
        ...

    @abstractmethod
    def _propose(
        self, x1: torch.Tensor, condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
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
        super().__init__(split_dims, d_context, net_kwargs)
        self.alpha = alpha

    def _build(self, net_kwargs):
        net_type = net_kwargs['net_type']

        # MLP Conditioner
        if net_type == 'mlp':
            net_kwargs.update({
                'd_input': self.split_dims[0] + self.d_context,
                'd_hidden': (net_kwargs['d_ff'],) * net_kwargs['depth'],
                'd_output': 2 * self.split_dims[1],
            })
            self.conditioner = FlowMLP(**net_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            net_kwargs.update({
                'd_input': self.split_dims[0],
                'd_context': self.d_context,
                'd_hidden': net_kwargs['d_ff'],
                'd_output': 2 * self.split_dims[1],
            })
            self.conditioner = FlowResidualNet(**net_kwargs)

    def _propose(self, x1, condition=None):
        parameters = self.conditioner(x1, condition)
        log_s, t = parameters.chunk(2, dim=-1)
        log_s = self.alpha * torch.tanh(log_s / self.alpha)  # softclamping
        return log_s, t

    def forward(self, x1, x2, condition=None, mask2=None, inverse=False):
        log_s, t = self._propose(x1, condition)
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        if inverse:
            x2 = (x2 - t) * (-log_s).exp()
        else:
            x2 = log_s.exp() * x2 + t
        log_det = log_s.sum(-1)
        return x2, log_det


