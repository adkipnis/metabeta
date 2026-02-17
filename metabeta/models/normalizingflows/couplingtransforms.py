from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metabeta.models.feedforward import FlowMLP, FlowResidualNet


class CouplingTransform(nn.Module):
    """Base class for coupling transforms:
    - build contexter network
    - propose transform parameters
    - directionally apply transform parameters to inputs
    """

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
    ) -> dict[str, torch.Tensor]:
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
        alpha: float = 1.0,  # softclamping scale
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.alpha = alpha

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
            subnet_kwargs.update(
                {
                    'd_input': 2 * self.split_dims[0] + self.d_context,
                    'd_hidden': (subnet_kwargs['d_ff'],) * subnet_kwargs['depth'],
                }
            )
            subnet_kwargs.pop('d_ff')
            subnet_kwargs.pop('depth')
            self.affine_conditioner = FlowMLP(**subnet_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            subnet_kwargs.update(
                {
                    'd_input': self.split_dims[0],
                    'd_context': self.split_dims[0] + self.d_context,
                    'd_hidden': subnet_kwargs['d_ff'],
                }
            )
            subnet_kwargs.pop('d_ff')
            self.affine_conditioner = FlowResidualNet(**subnet_kwargs)

    def _propose(self, x1, context=None, mask1=None):
        if mask1 is None:
            mask1 = torch.ones_like(x1)
        params = self.affine_conditioner(x1, context=context, mask=mask1)
        log_weight, bias = params.chunk(2, dim=-1)
        log_weight = self.alpha * torch.tanh(log_weight / self.alpha)   # softclamp
        return {'log_weight': log_weight, 'bias': bias}

    def forward(self, x1, x2, context=None, mask1=None, mask2=None, inverse=False):
        params = self._propose(x1, context, mask1)
        log_weight, bias = params['log_weight'], params['bias']
        if mask2 is not None:
            log_weight = log_weight * mask2
            bias = bias * mask2
        weight = log_weight.exp()
        if inverse:
            x2 = (x2 - bias) / weight
            log_det = -log_weight.sum(-1)
        else:
            x2 = weight * x2 + bias
            log_det = log_weight.sum(-1)
        return x2, log_det


class RationalQuadratic(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int,
        subnet_kwargs: dict | None = None,
        n_bins: int = 6,
        default_size: float = 1.0,
        min_bin: float = 0.1,
        min_deriv: float = 1e-3,
        adaptive_domain: bool = False,
        alpha: float = 1.0,  # softclamping
        eps: float = 1e-6,  # clamping
    ):
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context
        self.n_bins = n_bins
        self.default_left = -default_size
        self.default_right = default_size
        self.default_delta = default_size * 2
        self.min_delta = min_bin * n_bins
        self.min_bin = min_bin
        self.min_deriv = min_deriv
        self.adaptive_domain = adaptive_domain
        self.alpha = alpha
        self.eps = eps
        self._shift = np.log(np.e - 1)

        # set number of parameters per dim
        self.n_params_per_dim = 3 * self.n_bins - 1
        if self.adaptive_domain:
            self.n_params_per_dim += 4

        # check sizes
        if min_bin * n_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')

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
            subnet_kwargs.update(
                {
                    'd_input': 2 * self.split_dims[0] + self.d_context,
                    'd_hidden': (subnet_kwargs['d_ff'],) * subnet_kwargs['depth'],
                }
            )
            subnet_kwargs.pop('d_ff')
            subnet_kwargs.pop('depth')
            self.spline_conditioner = FlowMLP(**subnet_kwargs)

        # Residual Conditioner
        elif net_type == 'residual':
            subnet_kwargs.update(
                {
                    'd_input': self.split_dims[0],
                    'd_context': self.split_dims[0] + self.d_context,
                    'd_hidden': subnet_kwargs['d_ff'],
                }
            )
            subnet_kwargs.pop('d_ff')
            self.spline_conditioner = FlowResidualNet(**subnet_kwargs)

    def _propose(self, x1, context=None, mask1=None):
        if mask1 is None:
            mask1 = torch.ones_like(x1)
        params = self.spline_conditioner(x1, context=context, mask=mask1)
        params = params.reshape(*x1.shape[:-1], self.split_dims[1], -1)
        if self.n_params_per_dim != params.shape[-1]:
            raise ValueError(
                f'last params dim should be {self.n_params_per_dim} but found {params.shape[-1]}'
            )

        # unpack spline params
        k = self.n_bins
        widths = params[..., :k]
        heights = params[..., k : 2 * k]
        derivatives = params[..., 2 * k : 3 * k - 1]
        if self.adaptive_domain:
            log_weight = params[..., 3 * k - 1]
            bias = params[..., 3 * k]
            left = params[..., -2]
            delta_x = params[..., -1]
        else:
            log_weight = torch.zeros_like(params[..., 0])
            bias = torch.zeros_like(log_weight)
            left = torch.zeros_like(log_weight)
            delta_x = torch.zeros_like(log_weight)
        return {
            'widths': widths,
            'heights': heights,
            'derivatives': derivatives,
            'log_weight': log_weight,
            'bias': bias,
            'left': left.unsqueeze(-1),
            'delta_x': delta_x.unsqueeze(-1),
        }

    def _constrain(
        self,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        # --- affine params
        log_weight = params['log_weight']
        log_weight = self.alpha * torch.tanh(log_weight / self.alpha)   # softclamp
        weight = log_weight.exp()
        bias = params['bias']

        # --- derivatives
        derivatives = params['derivatives']
        derivatives = self.min_deriv + F.softplus(derivatives + self._shift)
        derivatives = F.pad(derivatives, (1, 1))
        derivatives[..., 0] = weight
        derivatives[..., -1] = weight

        # --- bounds
        left = self.default_left + params['left']
        delta_x = self.default_delta + params['delta_x']
        right = left + delta_x.clamp_min(self.min_delta)
        bottom = weight.unsqueeze(-1) * left + bias.unsqueeze(-1)
        top = weight.unsqueeze(-1) * right + bias.unsqueeze(-1)
        bounds = torch.cat([left, right, bottom, top], dim=-1)

        # --- widths
        # 1. normalize widths to sum to 1
        # 2. shift by min_val and ensure total_width sum to 1
        # 3. stretch to [left, right]
        widths = params['widths']
        widths = F.softmax(widths, dim=-1)
        widths = self.min_bin + (1 - self.min_bin * self.n_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0))
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left.squeeze(-1)
        cumwidths[..., -1] = right.squeeze(-1)

        # --- heights
        heights = params['heights']
        heights = F.softmax(heights, dim=-1)
        heights = self.min_bin + (1 - self.min_bin * self.n_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0))
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom.squeeze(-1)
        cumheights[..., -1] = top.squeeze(-1)

        return dict(
            bounds=bounds,
            cumwidths=cumwidths,
            cumheights=cumheights,
            derivatives=derivatives,
            log_weight=log_weight,
            bias=bias,
        )

    def forward(self, x1, x2, context=None, mask1=None, mask2=None, inverse=False):
        raw = self._propose(x1, context, mask1)
        params = self._constrain(raw)

        # construct boundary masks
        bounds = params['bounds'].chunk(4, -1)
        left, right, bottom, top = (t.squeeze(-1) for t in bounds)
        if inverse:
            inside = (bottom <= x2) & (x2 < top)
        else:
            inside = (left <= x2) & (x2 < right)
        outside = ~inside

        # incorporate mask
        if mask2 is not None:
            inside = inside & mask2.bool()
            outside = outside & mask2.bool()

        # init outputs
        log_det = torch.zeros_like(x2)
        z2 = x2.clone()

        # apply spline transform
        if inside.any():
            z2[inside], log_det[inside] = self._spline(x2, params, inside, inverse=inverse)

        # apply affine transform
        if outside.any():
            z2[outside], log_det[outside] = self._affine(x2, params, outside, inverse=inverse)
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
        cumwidths = params['cumwidths'][inside]
        cumheights = params['cumheights'][inside]
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
        heights = cumheights[..., 1:] - cumheights[..., :-1]
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
        log_weight = params['log_weight'][outside]
        bias = params['bias'][outside]

        # apply affine transform
        weight = log_weight.exp()
        if inverse:
            z2 = (x2 - bias) / weight
            log_det = -log_weight
        else:
            z2 = weight * x2 + bias
            log_det = log_weight
        return z2, log_det

    def _searchSorted(self, reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reference = reference.detach().clone()
        reference[..., -1] = reference[..., -1] + self.eps
        idx = torch.searchsorted(reference, target.unsqueeze(-1), right=True) - 1
        return idx


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seed = 0

    b = 512
    split_dims = (5, 1)
    d_context = 12
    n_bins = 6
    n_lines = 20
    lim = 15.2
    NET_KWARGS = {
        'net_type': 'mlp',
        'd_ff': 128,
        'depth': 2,
        'activation': 'ReLU',
        'zero_init': False,  # if True, the initial flows are identity maps
    }

    x2 = torch.linspace(-5, 5, b).unsqueeze(-1)

    # plot single affine
    torch.manual_seed(seed)
    plt.figure(figsize=(6, 6))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Affine')
    plt.grid(True)
    with torch.no_grad():
        for i in range(n_lines):
            transform = Affine(split_dims, d_context, NET_KWARGS)
            x1 = torch.randn(1, split_dims[0]).expand(b, -1)
            context = torch.randn(1, d_context).expand(b, -1)
            y2, _ = transform.forward(x1, x2, context)
            plt.plot(x2.numpy(), y2.numpy(), label=f'AC {i}', alpha=0.9)
    plt.plot(x2.numpy(), x2.numpy(), '--', alpha=0.5, label='identity')
    plt.ylim((-lim, lim))
    plt.show()

    # plot single spline
    torch.manual_seed(seed)

    plt.figure(figsize=(6, 6))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Rational Quadratic Spline')
    plt.grid(True)
    with torch.no_grad():
        for i in range(n_lines):
            transform = RationalQuadratic(
                split_dims, d_context, NET_KWARGS, n_bins=n_bins)
            x1 = torch.randn(1, split_dims[0]).expand(b, -1)
            context = torch.randn(1, d_context).expand(b, -1)
            y2, _ = transform.forward(x1, x2, context)
            plt.plot(x2.numpy(), y2.numpy(), label=f'NSF {i}', alpha=0.9)
    plt.plot(x2.numpy(), x2.numpy(), '--', alpha=0.5, label='identity')
    plt.ylim((-lim, lim))
    plt.show()
