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



class RationalQuadratic(CouplingTransform):
    # rewrite of code in https://github.com/bayesiains/nflows
    def __init__(
        self,
        # paramap 
        split_dims: List[int] | Tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
        num_bins: int = 10,
        tail_bound: float = 10.0,
        min_val: float = 1e-3,
        net_type: str = 'residual' # ['mlp', 'residual']
    ):
        super().__init__()
        self.d_out = split_dims[1]

        # MLP Paramap
        if net_type == 'mlp':
            kwargs = net_kwargs.copy()
            kwargs.update({
                'd_input': split_dims[0] + d_context,
                'd_output': split_dims[1] * (3 * num_bins - 1),
                'd_hidden': (net_kwargs['d_hidden'],) * net_kwargs['n_blocks'],
            })
            self.paramap = MLP(**kwargs)
            self.d_ff = kwargs['d_hidden'][-1]
        # Residual Paramap
        elif net_type == 'residual':
            net_kwargs.update({
                'd_input': split_dims[0],
                'd_output': split_dims[1] * (3 * num_bins - 1),
                'd_context': d_context,
            })
            self.paramap = ResidualNet(**net_kwargs)
            self.d_ff = net_kwargs['d_hidden']
        else:
            raise NotImplementedError(f'{net_type} must be either mlp or residual')

        # spline args
        if min_val * num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if min_val * num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_val
        self.min_bin_height = min_val
        self.min_derivative = min_val
        self.tail_constant = np.log(np.exp(1 - self.min_derivative) - 1)


    def _searchSorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations = bin_locations.detach().clone()
        bin_locations[..., -1] += eps
        idx = (torch.searchsorted(
            bin_locations, inputs[..., None],
            side="right",
        ) - 1).squeeze(-1)
        return idx 


    def propose(self, x1, condition=None):
        # get parameters
        parameters = self.paramap(x1, condition)
        parameters = parameters.reshape(*x1.shape[:-1], self.d_out, -1)

        # partition parameters
        k = self.num_bins
        raw_widths = parameters[..., :k] / np.sqrt(self.d_ff)
        raw_heights = parameters[..., k:2*k] / np.sqrt(self.d_ff)
        raw_derivs = parameters[..., 2*k:]
        return dict(raw_widths=raw_widths, raw_heights=raw_heights, raw_derivs=raw_derivs)


    def _constrain(self, raw_widths, raw_heights, raw_derivatives):
        # process widths
        left, right = -self.tail_bound, self.tail_bound # [-B, B]
        widths = F.softmax(raw_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths # ensure min
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0) # leftpad
        cumwidths = left + cumwidths * (right - left) # stretch cumulative widths to [-B, B]
        widths = cumwidths[..., 1:] - cumwidths[..., :-1] # adjust widths

        # process heights
        bottom, top = -self.tail_bound, self.tail_bound
        heights = F.softmax(raw_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = bottom + cumheights * (top - bottom)
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        # process derivatives
        derivatives = self.min_derivative + F.softplus(raw_derivatives)
        return widths, cumwidths, heights, cumheights, derivatives


    def __call__(self, input2, params, mask2=None, inverse=False):
        # extract params
        raw_widths, raw_heights, raw_derivs = params['raw_widths'], params['raw_heights'], params['raw_derivs']
        # masks
        inside_interval_mask = (input2 >= -self.tail_bound) & (input2 <= self.tail_bound)
        outside_interval_mask = ~inside_interval_mask

        # init outputs
        output2 = torch.zeros_like(input2)
        output2[outside_interval_mask] = input2[outside_interval_mask]
        log_det = torch.zeros_like(input2)

        # add linear tails
        raw_derivatives = F.pad(raw_derivs, pad=(1, 1))
        raw_derivatives[..., 0] = self.tail_constant
        raw_derivatives[..., -1] = self.tail_constant

        # apply spline inside of interval mask
        if torch.any(inside_interval_mask):
            f = self._inverse if inverse else self._forward
            output2_, log_det_ = f(
                input2[inside_interval_mask],
                raw_widths[inside_interval_mask, :],
                raw_heights[inside_interval_mask, :],
                raw_derivatives[inside_interval_mask, :]
                )
            output2[inside_interval_mask] = output2_
            log_det[inside_interval_mask] = log_det_

        # optionally mask outputs and logdet
        if mask2 is not None:
            output2 = output2 * mask2
            log_det = log_det * mask2
        return output2, log_det.sum(-1)


    def _forward(self, x2, raw_widths, raw_heights, raw_derivatives):
        ''' procedure based on [Durkan et al., 2019] '''

        if torch.min(x2) < -self.tail_bound or torch.max(x2) > self.tail_bound:
            raise ValueError('x2 outside of domain')

        # constrain parameters
        widths, cumwidths, heights, cumheights, derivatives = self._constrain(
            raw_widths, raw_heights, raw_derivatives)

        # get bin indices
        bin_idx = self._searchSorted(cumwidths, x2)[..., None]

        # for each knot k, get the rq parts
        x_k_delta = widths.gather(-1, bin_idx)[..., 0]
        x_k = cumwidths.gather(-1, bin_idx)[..., 0]
        y_k_delta = heights.gather(-1, bin_idx)[..., 0]
        y_k = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        s_k = delta.gather(-1, bin_idx)[..., 0]
        d_k = derivatives.gather(-1, bin_idx)[..., 0]
        d_k_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
        xi = (x2 - x_k) / x_k_delta
        xi_1_minus_xi = xi * (1 - xi)

        # construct rq splines
        alpha_k = y_k_delta * (s_k * xi.pow(2) + d_k * xi_1_minus_xi)
        beta_k = s_k + ((d_k_plus_one + d_k - 2 * s_k) * xi_1_minus_xi)
        z2 = y_k + alpha_k / beta_k

        # get log determinant
        derivative_numerator = s_k.pow(2) * (
            d_k_plus_one * xi.pow(2) + 2 * s_k * xi_1_minus_xi + d_k * (1 - xi).pow(2)
        )
        log_det = derivative_numerator.log() - 2 * beta_k.log()
        return z2, log_det


    def _inverse(self, z2, raw_widths, raw_heights, raw_derivatives):
        if torch.min(z2) < -self.tail_bound or torch.max(z2) > self.tail_bound:
            raise ValueError('z2 outside of domain')

        # constrain parameters
        widths, cumwidths, heights, cumheights, derivatives = self._constrain(
            raw_widths, raw_heights, raw_derivatives)

        # get bin indices
        bin_idx = self._searchSorted(cumheights, z2)[..., None]

        # for each knot k, get the rq parts
        x_k_delta = widths.gather(-1, bin_idx)[..., 0]
        x_k = cumwidths.gather(-1, bin_idx)[..., 0]
        y_k_delta = heights.gather(-1, bin_idx)[..., 0]
        y_k = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        s_k = delta.gather(-1, bin_idx)[..., 0]
        d_k = derivatives.gather(-1, bin_idx)[..., 0]
        d_k_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        # get analytical inverse of rq
        a = y_k_delta * (s_k - d_k) + (z2 - y_k) * (d_k_plus_one + d_k - 2 * s_k)
        b = y_k_delta * d_k - (z2 - y_k) * (d_k_plus_one + d_k - 2 * s_k)
        c = -s_k * (z2 - y_k)
        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()
        xi = (2 * c) / (-b - torch.sqrt(discriminant))
        x2 = xi * x_k_delta + x_k

        # get log determinant
        xi_1_minus_xi = xi * (1 - xi)
        beta_k = s_k + ((d_k_plus_one + d_k - 2 * s_k) * xi_1_minus_xi)
        derivative_numerator = s_k.pow(2) * (
            d_k_plus_one * xi.pow(2) + 2 * s_k * xi_1_minus_xi + d_k * (1 - xi).pow(2)
        )
        log_det = derivative_numerator.log() - 2 * beta_k.log()
        return x2, -log_det


