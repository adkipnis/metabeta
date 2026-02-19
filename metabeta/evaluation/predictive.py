from typing import Literal
import math
import torch
from torch import distributions as D
from metabeta.utils.evaluation import Proposal


def getPosteriorPredictive(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> D.Normal:
    """get p(y|X, theta)"""
    # parameters
    ffx = proposal.ffx   # (b, s, d)
    sigma_eps = proposal.sigma_eps   # (b, s)
    rfx = proposal.rfx   # (b, m, s, q)

    # observations
    X = data['X']   # (b, m, n, d)
    Z = data['Z']   # (b, m, n, q)

    # posterior params
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    loc = mu_g + mu_l
    scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
    return D.Normal(loc=loc, scale=scale)


def posteriorPredictiveNLL(
    pp: D.Normal,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
    mode: Literal['expected', 'mixture'] = 'mixture',
) -> torch.Tensor:
    """
        Returns per-dataset NLL with shape (b,).
        mode='expected':
            -E_p(theta|D)[ log p(y | theta) ]
        mode='mixture':
            -log E_p(theta|D)[ p(y | theta) ]
    """
    y = data['y'].unsqueeze(-1)   # (b, m, n, 1)
    log_p = pp.log_prob(y) # (b, m, n, s)
    obs_mask = data['mask_n'] # (b, m, n)
    n_obs = obs_mask.sum(dim=(1, 2)) # (b, )
    if mode == 'expected':
        log_p = log_p * obs_mask.unsqueeze(-1)
        nll_s = -log_p.sum(dim=(1, 2)) / n_obs.unsqueeze(-1) # (b, s)
        if w is None:
            return nll_s.mean(-1)
        return (nll_s * w).sum(-1)
    elif mode == 'mixture':
        if w is None:
            s = log_p.shape[-1]
            log_mix = torch.logsumexp(log_p, dim=-1) - math.log(s)
        else:
            log_w = w.log().unsqueeze(1).unsqueeze(1)
            log_mix = torch.logsumexp(log_p + log_w, dim=-1)
        return -(log_mix * obs_mask).sum(dim=(1,2)) / n_obs
    else:
        raise ValueError(f'unknown mode: {mode}')

def posteriorPredictiveSample(
    pp: D.Normal,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = data['mask_n'].unsqueeze(-1)
    y_rep = pp.sample((1,)).squeeze(0)
    y_rep = y_rep * mask
    return y_rep
