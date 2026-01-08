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


class RationalQuadratic(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
        n_bins: int = 16,
        tail_bound: float = 3.0,
        min_val: float = 1e-3,
        eps: float = 1e-6, # for clamping xi
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.min_val = min_val # bin width, height, derivative
        # self.d_ff = net_kwargs['d_ff'] # instead: zero init last layer of conditioner
        self.eps = eps
        self.tail_constant = np.log(np.exp(1 - self.min_val) - 1)
        assert n_bins * min_val < 1.0, 'either lower min_val or number of bins'

        # setup conditioner
        self._build(dict(net_kwargs))

    def _build(self, net_kwargs: dict):
        net_type = net_kwargs['net_type']
        assert net_type in ['mlp', 'residual']
        net_kwargs['d_output'] = (3 * self.n_bins + 1) * self.split_dims[1]

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
        params = self.conditioner(x1, condition)
        params = params.reshape(*x1.shape[:-1], self.split_dims[1], -1)
        k = self.n_bins
        widths = params[..., :k] #/ np.sqrt(self.d_ff)
        heights = params[..., k:2*k] #/ np.sqrt(self.d_ff)
        derivatives = params[..., 2*k:]
        # derivatives = F.pad(derivatives, (1,1))
        # derivatives[..., 0] = self.tail_constant
        # derivatives[..., -1] = self.tail_constant
        return widths, heights, derivatives

    def _constrain(
            self, params: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
        widths, heights, derivatives = params

        # setup bounds [-B, B]
        left = -self.tail_bound
        total_width = 2 * self.tail_bound
        bottom = -self.tail_bound
        total_height = 2 * self.tail_bound

        # normalize widths to sum to 1
        widths = F.softmax(widths, dim=-1)

        # shift by min_val but keep unit sum
        widths = self.min_val + (1 - self.n_bins * self.min_val) * widths

        # stretch to [-B, B]
        cumwidths = widths.cumsum(-1)
        cumwidths = F.pad(cumwidths, (1,0))
        cumwidths = left + total_width * cumwidths
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        # do the same with heights
        heights = F.softmax(heights, dim=-1)
        heights = self.min_val + (1 - self.n_bins * self.min_val) * heights
        cumheights = heights.cumsum(-1)
        cumheights = F.pad(cumheights, (1,0))
        cumheights = bottom + total_height * cumheights
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        # process derivatives
        derivatives = self.min_val + F.softplus(derivatives)
        return widths, cumwidths, heights, cumheights, derivatives

    def forward(self, x1, x2, condition=None, mask2=None, inverse=False):
        params = self._propose(x1, condition)

        # boundary masks
        inside = (x2 >= -self.tail_bound) & (x2 <= self.tail_bound)
        outside = ~inside
        if mask2 is not None:
            inside = inside & mask2.bool()
            outside = outside & mask2.bool()

        # init outputs
        log_det = torch.zeros_like(x2)
        z2 = x2.clone()

        # apply spline transform inside interval
        if torch.any(inside):
            z2[inside], log_det[inside] = self._spline(
                x2, params, inside, inverse=inverse)

        # optionally mask outputs
        if mask2 is not None:
            z2 = z2 * mask2
            log_det = log_det * mask2
        return z2, log_det.sum(-1)

    def _spline(
            self,
            x2: torch.Tensor,
            params: dict[str, torch.Tensor],
            inside: torch.Tensor,
            inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # constrain spline parameters
        x2 = x2[inside]
        params = {k: v[inside, :] for k,v in params.items()}
        widths, cumwidths, heights, cumheights, derivatives = self._constrain(params)

        # map each x to a bin between two knots
        if inverse:
            idx = self._searchSorted(cumheights, x2)
        else:
            idx = self._searchSorted(cumwidths, x2)
 
        # for each knot k, get the RQ parts
        x_k_delta = widths.gather(-1, idx)[..., 0]
        x_k = cumwidths.gather(-1, idx)[..., 0]
        y_k_delta = heights.gather(-1, idx)[..., 0]
        y_k = cumheights.gather(-1, idx)[..., 0]
        delta = heights / widths
        s_k = delta.gather(-1, idx)[..., 0]
        d_k_0 = derivatives.gather(-1, idx)[..., 0]
        d_k_1 = derivatives[..., 1:].gather(-1, idx)[..., 0]

        # apply RQ spline
        if inverse:
            # get analytical inverse of rq
            a = y_k_delta * (s_k - d_k_0) + (x2 - y_k) * (d_k_1 + d_k_0 - 2 * s_k)
            b = y_k_delta * d_k_0 - (x2 - y_k) * (d_k_1 + d_k_0 - 2 * s_k)
            c = -s_k * (x2 - y_k)
            discriminant = b.pow(2) - 4 * a * c
            xi = (2 * c) / (-b - torch.sqrt(discriminant))
            z2 = xi * x_k_delta + x_k

            # log_det variables
            xi_1_minus_xi = xi * (1 - xi)
            beta_k = s_k + ((d_k_1 + d_k_0 - 2 * s_k) * xi_1_minus_xi)
            ld_factor = -1
        else:
            # helper variables
            xi = (x2 - x_k) / x_k_delta
            xi_1_minus_xi = xi * (1 - xi)
            ld_factor = 1

            # construct RQ splines
            alpha_k = y_k_delta * (s_k * xi.pow(2) + d_k_0 * xi_1_minus_xi)
            beta_k = s_k + ((d_k_1 + d_k_0 - 2 * s_k) * xi_1_minus_xi)
            z2 = y_k + alpha_k / beta_k

        # get log determinant
        derivative_numerator = s_k.pow(2) * (
            d_k_1 * xi.pow(2) + 2 * s_k * xi_1_minus_xi + d_k_0 * (1 - xi).pow(2)
        )
        log_det = derivative_numerator.log() - 2 * beta_k.log()
        return z2, ld_factor * log_det

    def _searchSorted(
            self, reference: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
    ) -> torch.Tensor:
        reference = reference.detach().clone()
        reference[..., -1] += eps
        idx = torch.searchsorted(reference, target.unsqueeze(-1), right=True) - 1
        return idx


if __name__ == '__main__':
    b = 1
    split_dims = (3,2)
    d_context = 8

    NET_KWARGS = {
        'net_type': 'mlp',
        'd_ff': 128,
        'depth': 3,
        'activation': 'ReLU',
        'zero_init': False, # if True, the initial flows are identity maps
    }
    rq = RationalQuadratic(split_dims, d_context, NET_KWARGS, n_bins=4)

    # test raw params
    x1 = torch.randn(b, split_dims[0])
    x2 = torch.randn(b, split_dims[1])
    condition = torch.randn(b, d_context)
    params = rq._propose(x1, condition)
    rq._constrain(params)
    rq.forward(x1, x2, condition)
    
