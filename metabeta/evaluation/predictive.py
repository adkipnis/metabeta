from typing import Literal
import math
import torch
from torch import distributions as D

from metabeta.utils.evaluation import Proposal


def getPriorSamples(data: dict[str, torch.Tensor], n_samples: int) -> Proposal:
    shape = (n_samples,)
    out = {}

    # fixed effects
    mask = data['mask_d'].unsqueeze(-2)
    scale = data['tau_ffx'] + 1e-12
    ffx = D.Normal(0, scale).sample(shape).movedim(0, 1) * mask

    # sigma rfx
    mask = data['mask_q'].unsqueeze(-2)
    scale = data['tau_rfx'] + 1e-12
    sigma_rfx = D.HalfNormal(scale).sample(shape).movedim(0, 1) * mask   # type: ignore

    # sigma eps
    scale = data['tau_eps']
    sigma_eps = D.StudentT(df=4, scale=scale).sample(shape).movedim(0, 1)
    sigma_eps = sigma_eps.abs().unsqueeze(-1)

    # rfx
    mask = data['mask_mq'].unsqueeze(-2)
    shape = (int(data['m'].max()),)
    rfx = D.Normal(loc=0, scale=sigma_rfx + 1e-12).sample(shape).movedim(0, 1) * mask

    # bundle
    out['global'] = {'samples': torch.cat([ffx, sigma_rfx, sigma_eps], dim=-1)}
    out['local'] = {'samples': rfx}
    return Proposal(out)


def getPosteriorPredictive(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> D.Normal:
    """get p(y|X, theta)"""
    ffx = proposal.ffx  # (b, s, d)
    sigma_eps = proposal.sigma_eps  # (b, s)
    rfx = proposal.rfx  # (b, m, s, q)
    X = data['X']  # (b, m, n, d)
    Z = data['Z']  # (b, m, n, q)
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
        E_p(theta|D)[ -log p(y | theta) ]
    mode='mixture':
        -log E_p(theta|D)[ p(y | theta) ]
    """
    y_obs = data['y'].unsqueeze(-1)   # (b, m, n, 1)
    log_p = pp.log_prob(y_obs)   # (b, m, n, s)
    obs_mask = data['mask_n']   # (b, m, n)
    n_obs = obs_mask.sum(dim=(1, 2))   # (b, )
    if mode == 'expected':
        log_p = log_p * obs_mask.unsqueeze(-1)
        nll_s = -log_p.sum(dim=(1, 2)) / n_obs.unsqueeze(-1)   # (b, s)
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
        return -(log_mix * obs_mask).sum(dim=(1, 2)) / n_obs
    else:
        raise ValueError(f'unknown mode: {mode}')


def posteriorPredictiveSample(
    pp: D.Normal,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = data['mask_n'].unsqueeze(-1)
    y_rep = pp.sample((1,)).squeeze(0)
    y_rep = y_rep * mask
    return y_rep   # (b, m, n, s)


def intervalCheck(
    t_obs: torch.Tensor,  # (b, )
    t_rep: torch.Tensor,  # (b, s)
    alpha: float = 0.05,
) -> dict[str, torch.Tensor]:
    # TODO: incorporate importance weights
    lo = torch.quantile(t_rep, alpha / 2, dim=-1)
    hi = torch.quantile(t_rep, 1 - alpha / 2, dim=-1)
    outside = (t_obs < lo) | (hi < t_obs)
    left = (t_rep <= t_obs.unsqueeze(-1)).float().mean(-1)
    right = (t_rep >= t_obs.unsqueeze(-1)).float().mean(-1)
    tail_area = 2.0 * torch.minimum(left, right)
    return {
        'lo': lo,
        'hi': hi,
        'outside': outside,
        'outside_rate': outside.float().mean(),
        'tail_area': tail_area,
    }


def _groupMeans(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum(dim=2).clamp_min(1.0)   # (b, m, 1)
    return (y * mask).sum(dim=2) / n   # (b, m, s)


def ppcWithinGroupSD(y_rep: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
    mask_n = data['mask_n'].unsqueeze(-1).float()   # (b, m, n, 1)
    mask_m = data['mask_m'].unsqueeze(-1).float()   # (b, m, 1)
    means = _groupMeans(y_rep, mask_n)   # (b, m, s)
    n_i = data['ns'].clamp_min(1).unsqueeze(-1)   # (b, m, 1)
    var = ((y_rep - means.unsqueeze(2)).square() * mask_n).sum(dim=2) / n_i
    var_within = (var * mask_m).sum(dim=1) / data['m'].unsqueeze(-1)
    return var_within.sqrt()


def ppcBetweenGroupSD(y_rep: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
    mask_n = data['mask_n'].unsqueeze(-1).float()   # (b, m, n, 1)
    mask_m = data['mask_m'].unsqueeze(-1).float()   # (b, m, 1)
    means = _groupMeans(y_rep, mask_n)
    m = data['m'].unsqueeze(-1)
    mean = (means * mask_m).sum(dim=1) / m
    var_between = ((means - mean.unsqueeze(1)).square() * mask_m).sum(dim=1) / m
    return var_between.sqrt()


def ppcICC(sd_between: torch.Tensor, sd_within: torch.Tensor) -> torch.Tensor:
    var_between = sd_between.square()
    var_within = sd_within.square()
    return var_between / (var_between + var_within + 1e-12)
