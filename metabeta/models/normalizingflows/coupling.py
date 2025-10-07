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
    """
    learns an invertible conditional mapping from a target distribution to a standard normal
    mask indicates padding along the target dim (e.g. regression weights) and is dynamically propagated
    """

    def __init__(
        self,
        d_target: int,
        d_context: int = 0,
        n_flows: int = 6,
        permute_mode: str | None = "shuffle",
        use_actnorm: bool = True,
        use_linear: bool = True,
        base_type: str = "student",  # ['gaussian', 'student']
        transform: str = "affine",  # ['affine', 'spline']
        net_kwargs: dict = {},
    ):
        super().__init__()
        self.d_target = d_target

        # init base
        self.base_type = base_type
        Diags = {
            "gaussian": DiagGaussian,
            "normal": DiagGaussian,
            "student": DiagStudent,
        }
        if base_type in Diags:
            self.base = Diags[base_type](d_target)
        else:
            raise ValueError(f"desired base {base_type} not found")

        # init flows
        flows = []
        for _ in range(n_flows):
            if use_actnorm:
                flows.append(ActNorm(d_target))
            if use_linear:
                flows.append(LU(d_target, identity_init=True))
            if permute_mode is not None:
                flows.append(Permute(d_target, permute_mode))
            flows.append(
                DualCoupling(
                    d_data=d_target,
                    d_context=d_context,
                    transform=transform,
                    net_kwargs=net_kwargs,
                )
            )
        self.flows = nn.ModuleList(flows)

    def forward(self, x, condition=None, mask=None):
        # transform complex x to z in base dist
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        for flow in self.flows:
            x, ld, mask = flow.forward(x, condition, mask)
            # if x.isnan().any():
            #     print("nans in flow")
            log_det += ld
        return x, log_det, mask

    def inverse(self, z, condition=None, mask=None):
        # transform z in base dist to complex x
        log_det = torch.zeros(z.shape[:-1], device=z.device)
        for flow in reversed(self.flows):
            z, ld, mask = flow.inverse(z, condition, mask)  # type: ignore
            log_det += ld
        return z, log_det, mask

    def _forwardMask(self, mask: torch.Tensor) -> torch.Tensor:
        for flow in self.flows:
            mask = flow.forwardMask(mask)  # type: ignore
        return mask

    def _log_prob(
        self, z: torch.Tensor, log_det: torch.Tensor, mask=None
    ) -> torch.Tensor:
        log_probs = self.base.log_prob(z)
        if mask is not None:
            log_probs *= mask
        return log_probs.sum(dim=-1) + log_det

    def log_prob(self, x, condition=None, mask=None) -> torch.Tensor:
        z, log_det, mask = self.forward(x, condition, mask)
        return self._log_prob(z, log_det, mask)

    def sample(
        self,
        s: int = 100,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # determine shape
        shape = (1,)
        if context is not None:
            shape = context.shape[:-1]
        elif mask is not None:
            shape = mask.shape[:-1]
        sampling_shape = (*shape, s, self.d_target)

        # prepare context and mask
        if context is not None and context.dim() > 1:
            context = context.unsqueeze(-2).expand(*shape, s, -1)
        if mask is not None:
            if mask.dim() < len(sampling_shape):
                mask = mask.unsqueeze(-2).expand(*sampling_shape)
            mask_z = self._forwardMask(mask)
        else:
            mask_z = None

        # sample from base and optionally apply mask in base space
        z = self.base.sample(sampling_shape)
        if mask_z is not None:
            z = z * mask_z

        # project z back to x space
        x, log_det, _ = self.inverse(z, context, mask_z)

        # get probability in x space
        log_q = self._log_prob(z, log_det, mask_z)
        return x, log_q

