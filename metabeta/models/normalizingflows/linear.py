import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class Transform(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def inverse(
        self,
        z: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def forwardMask(self, mask: torch.Tensor):
        return mask


class ActNorm(Transform):
    # adapted from https://github.com/bayesflow/bayesflow to handle masking
    def __init__(self, target_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((target_dim,)))
        self.bias = nn.Parameter(torch.zeros((target_dim,)))

    def forward(self, x, condition=None, mask=None):
        s, t = self.scale.expand_as(x), self.bias.expand_as(x)
        if mask is not None:
            t = t * mask
        x = s * x + t
        log_det = s.abs().log()
        if mask is not None:
            log_det = log_det * mask
        return x, log_det.sum(-1), mask

    def inverse(self, z, condition=None, mask=None):
        s, t = self.scale.expand_as(z), self.bias.expand_as(z)
        # s, t = self.scale, self.bias
        if mask is not None:
            t = t * mask
        z = (z - t) / s
        log_det = -s.abs().log()
        if mask is not None:
            log_det = log_det * mask
        return z, log_det.sum(-1), mask


class Permute(Transform):
    # adapted from https://github.com/bayesflow/bayesflow to handle masking
    def __init__(self, target_dim: int, mode: str = "shuffle"):
        super().__init__()
        self.target_dim = target_dim
        assert mode in ["shuffle", "swap"], "unkown mode selected"
        self.mode = mode
        self.pivot = target_dim // 2
        self.inv_pivot = target_dim - self.pivot
        if self.mode == "shuffle":
            self.perm = torch.randperm(target_dim)
            self.inv_perm = self.perm.argsort()

    def forward(self, x, condition=None, mask=None):
        if self.mode == "shuffle":
            x = x[..., self.perm]
            if mask is not None:
                mask = mask[..., self.perm]
        elif self.mode == "swap":
            x1, x2 = x[..., : self.pivot], x[..., self.pivot :]
            x = torch.cat([x2, x1], dim=-1)
            if mask is not None:
                mask1, mask2 = mask[..., : self.pivot], mask[..., self.pivot :]
                mask = torch.cat([mask2, mask1], dim=-1)
        else:
            raise ValueError
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        return x, log_det, mask

    def forwardMask(self, mask):
        if self.mode == "shuffle":
            mask = mask[..., self.perm]
        elif self.mode == "swap":
            mask1, mask2 = mask[..., : self.pivot], mask[..., self.pivot :]
            mask = torch.cat([mask2, mask1], dim=-1)
        return mask

    def inverse(self, z, condition=None, mask=None):
        if self.mode == "shuffle":
            z = z[..., self.inv_perm]
            if mask is not None:
                mask = mask[..., self.inv_perm]
        elif self.mode == "swap":
            z1, z2 = z[..., : self.inv_pivot], z[..., self.inv_pivot :]
            z = torch.cat([z2, z1], dim=-1)
            if mask is not None:
                mask1, mask2 = mask[..., : self.inv_pivot], mask[..., self.inv_pivot :]
                mask = torch.cat([mask2, mask1], dim=-1)
        else:
            raise ValueError
        log_det = torch.zeros(z.shape[:-1], device=z.device)
        return z, log_det, mask


class LU(Transform):
    # rewrite of LU from https://github.com/bayesiains/nflows
    # implicit masking is done by setting the corresponding rows and columns of L and U to unit vectors
    def __init__(self, target_dim: int, identity_init: bool = True, eps: float = 1e-3):
        super().__init__()
        self.target_dim = target_dim
        self.eps = eps

        # indices for triangular matrices
        self.lower_indices = np.tril_indices(target_dim, k=-1)
        self.upper_indices = np.triu_indices(target_dim, k=1)
        self.diag_indices = np.diag_indices(target_dim)

        # learnable (but input-independent) parameters
        n_triangular_entries = ((target_dim - 1) * target_dim) // 2
        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(target_dim))
        self.bias = nn.Parameter(torch.zeros(target_dim))

        # init parameters
        nn.init.zeros_(self.bias)
        if identity_init:
            nn.init.zeros_(self.lower_entries)
            nn.init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            nn.init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(target_dim)
            nn.init.uniform_(self.lower_entries, -stdv, stdv)
            nn.init.uniform_(self.upper_entries, -stdv, stdv)
            nn.init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def _getLU(self, mask: torch.Tensor | None, b: int, d: int):
        device = self.bias.device

        # construct
        lower = torch.zeros(self.target_dim, self.target_dim, device=device)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = (
            1.0  # WLOG because the diagonal of U is can absorb the diagonal of L
        )
        upper = torch.zeros(self.target_dim, self.target_dim, device=device)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag
        # expand and mask
        lower = lower.unsqueeze(0).expand(b, d, d)
        upper = upper.unsqueeze(0).expand(b, d, d)
        bias = self.bias.unsqueeze(0).expand(b, d)
        if mask is not None:
            mask_ = mask.bool().unsqueeze(1) & mask.bool().unsqueeze(2)
            eye = torch.eye(d).expand(b, -1, -1).to(device)
            upper = torch.where(mask_, upper, eye)
            lower = torch.where(mask_, lower, eye)
            bias = bias * mask
        return lower, upper, bias

    def _forward(self, x: torch.Tensor, mask=None):
        lower, upper, bias = self._getLU(mask, *x.shape)
        z = torch.einsum("bij,bj->bi", upper, x)
        z = torch.einsum("bij,bj->bi", lower, z) + bias
        log_det = upper.diagonal(dim1=-2, dim2=-1).log()
        return z, log_det.sum(-1)

    def _inverse(self, z: torch.Tensor, mask=None):
        lower, upper, bias = self._getLU(mask, *z.shape)
        x = (z - bias).unsqueeze(-1)
        x = torch.linalg.solve_triangular(lower, x, upper=False)
        x = torch.linalg.solve_triangular(upper, x, upper=True).squeeze(-1)
        log_det = -upper.diagonal(dim1=-2, dim2=-1).log()
        return x, log_det.sum(-1)

    def __call__(self, inputs: torch.Tensor, mask=None, inverse=False):
        # wrapper for 2- and 3-dimensional inputs
        b, d = inputs.shape[0], inputs.shape[-1]
        if inputs.dim() > 2:
            inputs = inputs.reshape(-1, d)
            if mask is not None:
                mask = mask.reshape(-1, d)
        if inverse:
            outputs, log_det = self._inverse(inputs, mask)
        else:
            outputs, log_det = self._forward(inputs, mask)
        if outputs.shape[0] != b:
            outputs = outputs.reshape(b, -1, d)
            log_det = log_det.reshape(b, -1)
            if mask is not None:
                mask = mask.reshape(b, -1, d)
        return outputs, log_det, mask

    def forward(self, x, condition=None, mask=None):
        return self(x, mask=mask, inverse=False)

    def inverse(self, z, condition=None, mask=None):
        return self(z, mask=mask, inverse=True)


