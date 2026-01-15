from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Transform(nn.Module):
    ''' Base template for transforms used in normalizing flows '''
    @abstractmethod
    def forward(self, x: torch.Tensor,
              context: torch.Tensor | None = None,
              mask: torch.Tensor | None = None,
              inverse: bool = False,
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ...

    def inverse(self, x, context=None, mask=None):
        return self(x, context, mask, inverse=True)

    def _forwardMask(self, mask: torch.Tensor):
        return mask


class ActNorm(Transform):
    def __init__(self, d_target: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros((d_target,)))
        self.scale = nn.Parameter(torch.ones((d_target,)))
        self.register_buffer('initialized', torch.tensor(False))


    def _initialize(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        dims = tuple(range(x.dim() - 1))

        # get batch's moments
        if mask is not None:
            n = mask.sum(dims).clamp_min(1.0)
            mean = (x * mask).sum(dims) / n
            var = ((x - mean).square() * mask).sum(dims) / n
        else:
            mean = x.mean(dims)
            var = x.var(dims, unbiased=False)
        std = torch.sqrt(var + self.eps)

        # update params
        self.bias.data = -mean
        self.scale.data = 1.0 / std
        self.initialized.fill_(True) # type: ignore

    def forward(self, x, context=None, mask=None, inverse=False):
        if not self.initialized and not inverse:
            self._initialize(x, mask)
        scale, bias = self.scale.expand_as(x), self.bias.expand_as(x)
        if mask is not None:
            bias = bias * mask
        if inverse:
            x = (x - bias) / scale
        else:
            x = scale * x + bias
        log_det = scale.abs().log()
        if inverse:
            log_det = -log_det
        if mask is not None:
            log_det = log_det * mask
        return x, log_det.sum(-1), mask


class Permute(Transform):
    def __init__(self, d_target: int):
        super().__init__()
        self.d_target = d_target
        self.pivot = d_target // 2
        self.inv_pivot = d_target - self.pivot
        self.register_buffer('perm', torch.randperm(d_target))
        self.register_buffer('inv_perm', self.perm.argsort()) # type: ignore

    def forward(self, x, context=None, mask=None, inverse=False):
        perm: torch.Tensor = self.inv_perm if inverse else self.perm # type: ignore
        x = x[..., perm]
        if mask is not None:
            mask = mask[..., perm]
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        return x, log_det, mask

    def _forwardMask(self, mask):
        mask = mask[..., self.perm]
        return mask


class LU(Transform):
    ''' rewrite of LU from https://github.com/bayesiains/nflows
        implicit masking is done by setting the corresponding rows and columns of L and U to unit vectors
    '''
    def __init__(self, d_target: int, identity_init: bool = True, eps: float = 1e-3):
        super().__init__()
        self.d_target = d_target
        self.eps = eps

        # indices for triangular matrices
        self.lower_indices = np.tril_indices(d_target, k=-1)
        self.upper_indices = np.triu_indices(d_target, k=1)
        self.diag_indices = np.diag_indices(d_target)

        # learnable (but input-independent) parameters
        n_triangular_entries = ((d_target - 1) * d_target) // 2
        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(d_target))
        self.bias = nn.Parameter(torch.zeros(d_target))

        # init parameters
        nn.init.zeros_(self.bias)
        if identity_init:
            nn.init.zeros_(self.lower_entries)
            nn.init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            nn.init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(d_target)
            nn.init.uniform_(self.lower_entries, -stdv, stdv)
            nn.init.uniform_(self.upper_entries, -stdv, stdv)
            nn.init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def _getLU(self, mask: torch.Tensor | None, b: int, d: int):
        device = self.bias.device

        # construct
        lower = torch.zeros(self.d_target, self.d_target, device=device)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = (
            1.0  # WLOG because the diagonal of U is can absorb the diagonal of L
        )
        upper = torch.zeros(self.d_target, self.d_target, device=device)
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

    def forward(self, x, context=None, mask=None, inverse=False):
        # handle 3D inputs
        b, *_, d = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, d)
            if mask is not None:
                mask = mask.reshape(-1, d)
        elif x.dim() > 3:
            raise NotImplementedError

        # multiply x with invertible matrix
        lower, upper, bias = self._getLU(mask, *x.shape)
        log_det = upper.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        if inverse:
            x = (x - bias).unsqueeze(-1)
            x = torch.linalg.solve_triangular(lower, x, upper=False)
            x = torch.linalg.solve_triangular(upper, x, upper=True).squeeze(-1)
            log_det = -log_det
        else:
            x = torch.einsum("bij,bj->bi", upper, x)
            x = torch.einsum("bij,bj->bi", lower, x) + bias

        # reshape back if 3D
        if len(x) != b:
            x = x.reshape(b, -1, d)
            log_det = log_det.reshape(b, -1)
            if mask is not None:
                mask = mask.reshape(b, -1, d)
        return x, log_det, mask

