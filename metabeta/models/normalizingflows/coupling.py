from typing import List, Tuple
import torch
from torch import nn
from metabeta.models.normalizingflows.linear import Transform, Permute, LU, ActNorm
from metabeta.models.normalizingflows.couplingtransforms import Affine, RationalQuadratic
from metabeta.models.normalizingflows.distributions import DiagGaussian, DiagStudent


class Coupling(nn.Module):
    def __init__(
        self,
        split_dims: List[int] | Tuple[int, int],
        d_context: int = 0,
        transform: str = "affine",
        net_kwargs: dict = {},
    ):
        super().__init__()
        if transform == "affine":
            self.transform = Affine(
                split_dims=split_dims, d_context=d_context, net_kwargs=net_kwargs
            )
        elif transform == "spline":
            self.transform = RationalQuadratic(
                split_dims=split_dims, d_context=d_context, net_kwargs=net_kwargs
            )
        else:
            raise NotImplementedError()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, condition=None, mask2=None):
        parameters = self.transform.propose(x1, condition)
        x2, log_det = self.transform.forward(x2, parameters, mask2=mask2)
        return (x1, x2), log_det

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor, condition=None, mask2=None):
        parameters = self.transform.propose(z1, condition)
        z2, log_det = self.transform.inverse(z2, parameters, mask2=mask2)
        return (z1, z2), log_det


class DualCoupling(Transform):
    def __init__(
        self,
        d_data: int,
        d_context: int = 0,
        transform: str = "affine",
        net_kwargs: dict = {},
    ):
        super().__init__()
        self.pivot = d_data // 2
        split_dims = [self.pivot, d_data - self.pivot]
        self.coupling1 = Coupling(
            split_dims=split_dims,
            d_context=d_context,
            net_kwargs=net_kwargs,
            transform=transform,
        )
        self.coupling2 = Coupling(
            split_dims=split_dims[::-1],
            d_context=d_context,
            net_kwargs=net_kwargs,
            transform=transform,
        )

    def forwardMask(self, mask: torch.Tensor):
        return mask

    def forward(self, x, condition=None, mask=None):
        z = torch.empty_like(x)
        x1, x2 = x[..., : self.pivot], x[..., self.pivot :]
        if mask is not None:
            mask1, mask2 = mask[..., : self.pivot], mask[..., self.pivot :]
        else:
            mask1, mask2 = None, None
        (x1, x2), log_det1 = self.coupling1(x1, x2, condition, mask2)
        (x2, x1), log_det2 = self.coupling2(x2, x1, condition, mask1)
        z[..., : self.pivot], z[..., self.pivot :] = x1, x2
        log_det = log_det1 + log_det2
        return z, log_det, mask

    def inverse(self, z, condition=None, mask=None):
        x = torch.empty_like(z)
        z1, z2 = z[..., : self.pivot], z[..., self.pivot :]
        if mask is not None:
            mask1, mask2 = mask[..., : self.pivot], mask[..., self.pivot :]
        else:
            mask1, mask2 = None, None
        (z2, z1), log_det2 = self.coupling2.inverse(z2, z1, condition, mask1)
        (z1, z2), log_det1 = self.coupling1.inverse(z1, z2, condition, mask2)
        x[..., : self.pivot], x[..., self.pivot :] = z1, z2
        log_det = log_det1 + log_det2
        return x, log_det, mask


class CouplingFlow(nn.Module):
