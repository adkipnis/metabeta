from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.feedforward import FlowMLP, FlowResidualNet


class CouplingTransform(nn.Module):
    ''' Base class for coupling transforms:
        - build contexter network
        - propose transform parameters
        - directionally apply transform parameters to inputs
    '''

    SUBNET_KWARGS = {
        'net_type': 'mlp',
        'd_ff': 128,
        'depth': 3,
    }

    @abstractmethod
    def _build(self, subnet_kwargs: dict) -> None:
        ...


    @abstractmethod
    def _propose(
        self,
        x1: torch.Tensor,
        context: torch.Tensor | None = None,
        mask1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        ...

    @abstractmethod
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        context: torch.Tensor | None = None,
        mask1: torch.Tensor | None = None,
        mask2: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...

    def inverse(self, x1, x2, context=None, mask1=None, mask2=None):
        return self(x1, x2, context, mask1, mask2, inverse=True)


class Affine(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int,
        subnet_kwargs: dict | None = None,
        alpha: float = 1.5, # softclamping scale
        beta: float = 5.0, # softclamping bias
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.alpha = alpha
        self.beta = beta

        # safely handle subnet_cfg
        if subnet_kwargs is None:
            subnet_kwargs = {}
        subnet_kwargs = self.SUBNET_KWARGS | subnet_kwargs

        # setup conditioner
        self._build(subnet_kwargs)

    def _build(self, subnet_kwargs):
        net_type = subnet_kwargs.pop('net_type')
        assert net_type in ['mlp', 'residual']
        subnet_kwargs['d_output'] = 2 * self.split_dims[1]
        
        # MLP Conditioner
        if net_type == 'mlp':
            subnet_kwargs.update({
                'd_input': 2 * self.split_dims[0] + self.d_context,
                'd_hidden': (subnet_kwargs['d_ff'],) * subnet_kwargs['depth'],
            })
            subnet_kwargs.pop('d_ff')
            subnet_kwargs.pop('depth')
            self.conditioner = FlowMLP(**subnet_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            subnet_kwargs.update({
                'd_input': self.split_dims[0],
                'd_context': self.split_dims[0] + self.d_context,
                'd_hidden': subnet_kwargs['d_ff'],
            })
            subnet_kwargs.pop('d_ff')
            self.conditioner = FlowResidualNet(**subnet_kwargs)
    
    def _propose(self, x1, context=None, mask1=None):
        if mask1 is None:
            mask1 = torch.ones_like(x1)
        params = self.conditioner(x1, context=context, mask=mask1)
        log_s, t = params.chunk(2, dim=-1)
        # softclamp
        log_s = self.alpha * torch.tanh(log_s / self.alpha)
        t = self.beta * torch.tanh(t / self.beta)
        return log_s, t

    def forward(self, x1, x2, context=None, mask1=None, mask2=None, inverse=False):
        log_s, t = self._propose(x1, context, mask1)
        if mask2 is not None:
            log_s = log_s * mask2
            t = t * mask2
        s = log_s.exp()
        if inverse:
            x2 = (x2 - t) / s
        else:
            x2 = s * x2 + t
        log_det = log_s.sum(-1)
        return x2, log_det


class RationalQuadratic(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int,
        subnet_kwargs: dict | None = None,
        n_bins: int = 8,
        default_size: float = 2.0,
        min_total: float = 0.5,
        min_rel: float = 0.5,
        max_rel: float = 1.0,
        min_bin: float = 0.1,
        eps: float = 1e-6, # for clamping
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.n_bins = n_bins
        self.n_params_per_dim = (3 * self.n_bins + 3)
        self.default_left = -default_size
        self.default_bottom = -default_size
        self.default_width = 2 * default_size
        self.default_height = 2 * default_size
        self.min_total = min_total
        self.min_rel = min_rel
        self.max_rel = max_rel
        self.min_bin = min_bin
        self.eps = eps
        self._shift = np.log(np.e - 1)

        # safely handle subnet_cfg
        if subnet_kwargs is None:
            subnet_kwargs = {}
        subnet_kwargs = self.SUBNET_KWARGS | subnet_kwargs

        # setup conditioner
        self._build(subnet_kwargs)

    def _build(self, subnet_kwargs):
        net_type = subnet_kwargs.pop('net_type')
        assert net_type in ['mlp', 'residual']
        subnet_kwargs['d_output'] = self.n_params_per_dim * self.split_dims[1]

        # MLP Conditioner
        if net_type == 'mlp':
            subnet_kwargs.update({
                'd_input': 2 * self.split_dims[0] + self.d_context,
                'd_hidden': (subnet_kwargs['d_ff'],) * subnet_kwargs['depth'],
            })
            subnet_kwargs.pop('d_ff')
            subnet_kwargs.pop('depth')
            self.conditioner = FlowMLP(**subnet_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            subnet_kwargs.update({
                'd_input': self.split_dims[0],
                'd_context': self.split_dims[0] + self.d_context,
                'd_hidden': subnet_kwargs['d_ff'],
            })
            subnet_kwargs.pop('d_ff')
            self.conditioner = FlowResidualNet(**subnet_kwargs)

    def _propose(self, x1, context=None, mask1=None):
        if mask1 is None:
            mask1 = torch.ones_like(x1)
        params = self.conditioner(x1, context=context, mask=mask1)
        params = params.reshape(*x1.shape[:-1], self.split_dims[1], -1)
        if self.n_params_per_dim != params.shape[-1]:
            raise ValueError(
                f'last params dim should be {self.n_params_per_dim} but found {params.shape[-1]}')
        k = self.n_bins
        widths = params[..., :k]
        heights = params[..., k:2*k]
        derivatives = params[..., 2*k:-4]
        bounds = params[..., -4:]
        return bounds, widths, heights, derivatives

    def _constrain(
            self, params: tuple[torch.Tensor, ...],
            ) -> dict[str, torch.Tensor]:
        bounds, widths, heights, derivatives = params
        left, total_width, bottom, total_height = bounds.chunk(4, -1)

        # --- bounds
        # lower bounds
        left = left + self.default_left
        bottom = bottom + self.default_bottom

        # upper bounds (individually scale default total width resp. height)
        relative_width = torch.asinh(F.softplus(total_width + self._sin_shift))
        total_width = self.min_total + (self.default_width - self.min_total) * relative_width
        relative_height = torch.asinh(F.softplus(total_height + self._sin_shift))
        total_height = self.min_total + (self.default_height - self.min_total) * relative_height

        bounds = torch.cat([left, total_width, bottom, total_height], dim=-1)

        # --- bins
        # normalize widths to sum to 1
        widths = F.softmax(widths, dim=-1)

        # shift by min_val and ensure total_width sum
        widths = self.min_bin + (total_width - self.n_bins * self.min_bin) * widths

        # stretch to [left, left + total_width]
        cumwidths = widths.cumsum(-1)
        cumwidths = left + F.pad(cumwidths, (1,0))
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        # do the same with heights
        heights = F.softmax(heights, dim=-1)
        heights = self.min_bin + (total_height - self.n_bins * self.min_bin) * heights
        cumheights = heights.cumsum(-1)
        cumheights = bottom + F.pad(cumheights, (1,0))
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        # --- affine params
        weight = total_height / total_width
        bias = bottom - weight * left
        weight = weight.squeeze(-1)
        bias = bias.squeeze(-1)

        # --- derivatives
        derivatives = 1e-3 + F.softplus(derivatives)
        derivatives = F.pad(derivatives, (1,1))
        derivatives[..., 0] = derivatives[..., -1] = weight

        return dict(
            bounds=bounds,
            widths=widths,
            cumwidths=cumwidths,
            heights=heights,
            cumheights=cumheights,
            weight=weight,
            bias=bias,
            derivatives=derivatives,
        )

    def forward(self, x1, x2, context=None, mask2=None, inverse=False):
        raw = self._propose(x1, context)
        params = self._constrain(raw)

        # construct boundary masks
        bounds = params['bounds'].chunk(4, -1)
        left, total_width, bottom, total_height = (t.squeeze(-1) for t in bounds)
        if inverse:
            top = bottom + total_height
            inside = (bottom <= x2) & (x2 <= top)
        else:
            right = left + total_width
            inside = (left <= x2) & (x2 <= right)
        outside = ~inside

        # incorporate mask
        if mask2 is not None:
            inside = inside & mask2.bool()
            outside = outside & mask2.bool()

        # init outputs
        log_det = torch.zeros_like(x2)
        z2 = torch.zeros_like(x2)

        # apply spline transform
        z2[inside], log_det[inside] = self._spline(
            x2, params, inside, inverse=inverse)

        # apply affine transform
        z2[outside], log_det[outside] = self._affine(
            x2, params, outside, inverse=inverse)
        return z2, log_det.sum(-1)

    def _spline(
            self,
            x2: torch.Tensor,
            params: dict[str, torch.Tensor],
            inside: torch.Tensor,
            inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # unpack and apply mask
        x2 = x2[inside]
        widths = params['widths'][inside]
        cumwidths = params['cumwidths'][inside]
        heights = params['heights'][inside]
        cumheights = params['cumheights'][inside]
        derivatives = params['derivatives'][inside]

        # map each x to a bin between two knots
        if inverse:
            idx = self._searchSorted(cumheights, x2)
        else:
            idx = self._searchSorted(cumwidths, x2)
 
        # for each knot k, get the RQ parts
        x_k_delta = widths.gather(-1, idx).squeeze(-1)
        x_k = cumwidths.gather(-1, idx).squeeze(-1)
        y_k_delta = heights.gather(-1, idx).squeeze(-1)
        y_k = cumheights.gather(-1, idx).squeeze(-1)
        delta = heights / widths
        s_k = delta.gather(-1, idx).squeeze(-1)
        d_k_0 = derivatives.gather(-1, idx).squeeze(-1)
        d_k_1 = derivatives[..., 1:].gather(-1, idx).squeeze(-1)

        # apply RQ spline
        if inverse:
            # get analytical inverse of rq
            a = y_k_delta * (s_k - d_k_0) + (x2 - y_k) * (d_k_1 + d_k_0 - 2 * s_k)
            b = y_k_delta * d_k_0 - (x2 - y_k) * (d_k_1 + d_k_0 - 2 * s_k)
            c = -s_k * (x2 - y_k)
            discriminant = b.pow(2) - 4 * a * c
            discriminant = torch.clamp(discriminant, min=1e-12)
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

    def _affine(
            self,
            x2: torch.Tensor,
            params: dict[str, torch.Tensor],
            outside: torch.Tensor,
            inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # unpack and apply mask
        x2 = x2[outside]
        weight = params['weight'][outside]
        bias = params['bias'][outside]
        log_det = weight.log()

        # apply affine transform
        if inverse:
            ld_factor = -1
            z2 = (x2 - bias) / weight
        else:
            ld_factor = 1
            z2 = weight * x2 + bias
        return z2, ld_factor * log_det


    def _searchSorted(
            self, reference: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
    ) -> torch.Tensor:
        reference = reference.detach() # bin-assignment is not differentiable
        reference[..., -1] += eps
        idx = torch.searchsorted(reference, target.unsqueeze(-1), right=True) - 1
        return idx



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    torch.manual_seed(0)

    b = 1
    split_dims = (5,3)
    d_context = 12
    n_bins = 16

    NET_KWARGS = {
        'net_type': 'mlp',
        'd_ff': 128,
        'depth': 3,
        'activation': 'ReLU',
        'zero_init': False, # if True, the initial flows are identity maps
    }
    rq = RationalQuadratic(split_dims, d_context, NET_KWARGS, n_bins=n_bins)

    # test raw params
    x1 = torch.randn(b, split_dims[0])
    x2 = torch.randn(b, split_dims[1]) * 3
    context = torch.randn(b, d_context)
    params = rq._propose(x1, context)
    rq._constrain(params)
    rq.forward(x1, x2, context)

    # plot single spline
    with torch.no_grad():
        n_bins=16
        b = 512
        split_dims = (5,1)
        rq = RationalQuadratic(split_dims, d_context, NET_KWARGS, n_bins=n_bins)
        x1 = torch.randn(1, split_dims[0]).expand(b, -1)
        x2 = torch.linspace(-6, 6, b).unsqueeze(-1)
        context = torch.randn(1, d_context).expand(b, -1)
        y2, _ = rq.forward(x1, x2, context)

        # Plot
        plt.figure(figsize=(6, 6))
        plt.plot(x2.numpy(), y2.numpy(), label="RQ spline")
        plt.plot(x2.numpy(), x2.numpy(), "--", alpha=0.5, label="identity")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Rational Quadratic Spline")
        plt.legend()
        plt.grid(True)
        plt.show()
