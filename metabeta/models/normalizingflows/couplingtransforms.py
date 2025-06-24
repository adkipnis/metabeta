from typing import List, Tuple, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.feedforward import MLP, ResidualNet


class CouplingTransform(nn.Module):
    def propose(self,
                x1: torch.Tensor,
                condition: torch.Tensor|None=None
                ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self,
                x2: torch.Tensor,
                params: Dict[str, torch.Tensor],
                mask2: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError

    def inverse(self,
                z2: torch.Tensor,
                params: Dict[str, torch.Tensor],
                mask2: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError


class Affine(CouplingTransform):
    def __init__(self,
                 split_dims: List[int] | Tuple[int, int],
                 d_context: int = 0,
                 net_kwargs: dict = {},
                 net_type: str = 'residual' # ['mlp', 'residual']
                ):
        super().__init__()

        # MLP Paramap
        if net_type == 'mlp':
            kwargs = net_kwargs.copy()
            kwargs.update({
                'd_input': split_dims[0] + d_context,
                'd_output': 2 * split_dims[1],
                'd_hidden': (net_kwargs['d_hidden'],) * net_kwargs['n_blocks'],
            })
            self.paramap = MLP(**kwargs)

        # Residual Paramap
        elif net_type == 'residual':
            net_kwargs.update({
                'd_input': split_dims[0],
                'd_output': 2 * split_dims[1],
                'd_context': d_context,
            })
            self.paramap = ResidualNet(**net_kwargs)

        else: 
            raise NotImplementedError(f'{net_type} must be either mlp or residual')

    def propose(self, x1, condition=None):
        parameters = self.paramap(x1, condition)
        log_s, t = parameters.chunk(2, dim=-1)
        log_s = torch.arcsinh(log_s) # softclamping
        return dict(log_s=log_s, t=t)
 
    def forward(self, x2, params, mask2=None):
        log_s, t = params['log_s'], params['t']
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        z2 = log_s.exp() * x2 + t
        log_det = log_s.sum(-1)
        return z2, log_det

    def inverse(self, z2, params, mask2=None):
        log_s, t = params['log_s'], params['t']
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        x2 = (z2 - t) * (-log_s).exp()
        log_det = -log_s.sum(-1)
        return x2, log_det



