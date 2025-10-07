from typing import List, Tuple, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.feedforward import MLP, ResidualNet


class CouplingTransform(nn.Module):
    def propose(
        self, x1: torch.Tensor, condition: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        x2: torch.Tensor,
        params: Dict[str, torch.Tensor],
        mask2: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def inverse(
        self,
        z2: torch.Tensor,
        params: Dict[str, torch.Tensor],
        mask2: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


class Affine(CouplingTransform):
    def __init__(
        self,
        split_dims: List[int] | Tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
    ):
        super().__init__()
        net_type = net_kwargs.get("net_type", "mlp")  # ['mlp', 'residual']

        # MLP Conditioner
        if net_type == "mlp":
            kwargs = net_kwargs.copy()
            kwargs.update(
                {
                    "d_input": split_dims[0] + d_context,
                    "d_output": 2 * split_dims[1],
                    "d_hidden": (net_kwargs["d_ff"],) * net_kwargs["depth"],
                }
            )
            self.paramap = MLP(**kwargs)

        # Residual Conditioner
        elif net_type == "residual":
            net_kwargs.update(
                {
                    "d_input": split_dims[0],
                    "d_output": 2 * split_dims[1],
                    "d_context": d_context,
                }
            )
            self.paramap = ResidualNet(**net_kwargs)
        else:
            raise NotImplementedError(f"{net_type} must be either MLP or ResidualNet")

        # Set last layer to almost zeros
        last_layer = [m for m in self.paramap.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_layer.weight, -1e-3, 1e-3)

    def propose(self, x1, condition=None, alpha: float = 1.0):
        parameters = self.paramap(x1, condition)
        log_s, t = parameters.chunk(2, dim=-1)
        log_s = alpha * torch.tanh(log_s / alpha)  # softclamping
        return dict(log_s=log_s, t=t)

    def forward(self, x2, params, mask2=None):
        log_s, t = params["log_s"], params["t"]
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        z2 = log_s.exp() * x2 + t
        log_det = log_s.sum(-1)
        return z2, log_det

    def inverse(self, z2, params, mask2=None):
        log_s, t = params["log_s"], params["t"]
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        x2 = (z2 - t) * (-log_s).exp()
        log_det = log_s.sum(-1)
        return x2, -log_det


class RationalQuadratic(CouplingTransform):
