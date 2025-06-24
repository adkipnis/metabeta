from typing import List, Tuple
import torch
from torch import nn
from metabeta.models.normalizingflows.linear import Transform, Permute, LU, ActNorm
from metabeta.models.normalizingflows.couplingtransforms import Affine, RationalQuadratic
from metabeta.models.normalizingflows.distributions import DiagGaussian, DiagStudent, DiagUniform


class Coupling(nn.Module):
    def __init__(self,
                 split_dims: List[int] | Tuple[int, int],
                 d_context: int = 0,
                 transform: str = 'affine',
                 net_kwargs: dict = {},
                ):
        super().__init__()
        if transform == 'affine':
            self.transform = Affine(
                split_dims=split_dims, d_context=d_context, net_kwargs=net_kwargs)
        elif transform == 'rq':
            self.transform = RationalQuadratic(
                split_dims=split_dims, d_context=d_context, net_kwargs=net_kwargs)
        else:
            raise NotImplementedError()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, condition=None, mask2=None):
        z1 = x1
        parameters = self.transform.propose(x1, condition)
        z2, log_det = self.transform.forward(x2, parameters, mask2=mask2)
        return (z1, z2), log_det

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor, condition=None, mask2=None):
        x1 = z1
        parameters = self.transform.propose(z1, condition)
        x2, log_det = self.transform.inverse(z2, parameters, mask2=mask2)
        return (x1, x2), log_det


