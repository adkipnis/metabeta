from typing import Literal
import warnings
import math
import numpy as np
import arviz as az
from scipy.special import logsumexp as sp_logsumexp
import torch
from torch import distributions as D

from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import (
    hasSigmaEps,
    posteriorPredictiveDist,
    sampleFfxTorch,
    sampleSigmaTorch,
)


def getPriorSamples(
    data: dict[str, torch.Tensor], n_samples: int, likelihood_family: int = 0
) -> Proposal:
    has_eps = hasSigmaEps(likelihood_family)
    shape = (n_samples,)

    # fixed effects
    mask = data['mask_d'].unsqueeze(-2)
    loc, scale = data['nu_ffx'], data['tau_ffx'] + 1e-12
    family_ffx = data['family_ffx']
    ffx = sampleFfxTorch(loc, scale, family_ffx, shape).movedim(0, 1) * mask

    # sigma rfx
    mask = data['mask_q'].unsqueeze(-2)
    scale = data['tau_rfx'] + 1e-12
    family_sigma_rfx = data['family_sigma_rfx']
    sigma_rfx = sampleSigmaTorch(scale, family_sigma_rfx, shape).movedim(0, 1) * mask

    # sigma eps
    global_parts = [ffx, sigma_rfx]
    if has_eps:
        scale = data['tau_eps'] + 1e-12
        family_sigma_eps = data['family_sigma_eps']
        sigma_eps = sampleSigmaTorch(scale, family_sigma_eps, shape).movedim(0, 1)
        global_parts.append(sigma_eps.unsqueeze(-1))

    # rfx
    mask = data['mask_mq'].unsqueeze(-2)
    # shape = (int(data['m'].max()),)
    shape = (mask.shape[1],)
    rfx = D.Normal(loc=0, scale=sigma_rfx + 1e-12).sample(shape).movedim(0, 1) * mask

    # bundle
    out = {
        'global': {'samples': torch.cat(global_parts, dim=-1)},
        'local': {'samples': rfx},
    }
    return Proposal(out, has_sigma_eps=has_eps)


def getPosteriorPredictive(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    likelihood_family: int = 0,
) -> D.Distribution:
    """get p(y|X, theta)"""
    ffx = proposal.ffx  # (b, s, d)
    sigma_eps = proposal.sigma_eps if proposal.has_sigma_eps else ffx.new_zeros(ffx.shape[:2])
    rfx = proposal.rfx  # (b, m, s, q)
    return posteriorPredictiveDist(ffx, sigma_eps, rfx, data['X'], data['Z'], likelihood_family)


def posteriorPredictiveNLL(
    pp: D.Distribution,
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


def psisLooNLL(
    pp: D.Distribution,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PSIS-LOO NLL per dataset. Returns (loo_nll, pareto_k), both shape (b,).

    Leaves out one within-group observation at a time and uses Pareto-smoothed
    importance sampling (PSIS) to stabilize the harmonic-mean LOO estimator.

    The LOO IS weight for observation (i, j) and sample s is
        log w_ij^s = -log p(y_ij | θ^s)   [+ log w_IS^s if w is provided]
    PSIS smooths the upper tail of these weights, then the LOO-predictive
    log-density is  log p_loo(y_ij) = logsumexp(log_w_psis_norm + log_p, dim=-1)
    where log_w_psis_norm are the log-normalized PSIS weights.

    When PSIS fails for an observation (degenerate weights, too few samples),
    the raw normalized log-weights are used as a fallback and pareto_k is nan.

    Parameters
    ----------
    w : existing IS weights (b, s). When provided, folds in the IS correction
        so that both the proposal-posterior gap and the LOO leave-out are
        handled jointly: log w_ij^s = log w_IS^s - log p(y_ij | θ^s).
        When None, the raw proposal is treated as the posterior (uniform base).
    """

    y_obs = data['y'].unsqueeze(-1)   # (b, m, n, 1)
    log_p = pp.log_prob(y_obs)        # (b, m, n, s)
    mask = data['mask_n']             # (b, m, n)
    n_obs = mask.sum(dim=(1, 2))      # (b,)

    # LOO IS log-weights: log w_ij^s = -log p(y_ij | θ^s) [+ log w_IS^s]
    log_w = -log_p                    # (b, m, n, s)
    if w is not None:
        log_w = log_w + w.log().unsqueeze(1).unsqueeze(1)

    b, m, n, s = log_p.shape

    # PSIS per observation — az.psislw expects (n_obs, n_draws) with last axis = samples.
    # Samples are i.i.d. from the flow, so reff=1.
    # Only run PSIS on real (non-padded) observations; scatter results back afterward.
    valid = mask.reshape(b * m * n).numpy()  # (b*m*n,) bool
    log_w_flat = log_w.detach().reshape(b * m * n, s).numpy()
    log_w_valid = log_w_flat[valid]          # (n_valid, s)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        log_w_psis_valid, k_valid = az.psislw(log_w_valid, reff=1.0)

    # Fallback for rows where PSIS fails (degenerate / constant weights):
    # substitute normalized raw log-weights and flag k as nan.
    bad = ~np.isfinite(log_w_psis_valid).all(axis=-1)
    if bad.any():
        raw_norm = log_w_valid[bad] - sp_logsumexp(log_w_valid[bad], axis=-1, keepdims=True)
        log_w_psis_valid[bad] = raw_norm
        k_valid[bad] = np.nan

    # Also replace inf k with nan so they don't dominate means/medians.
    k_valid = np.where(np.isfinite(k_valid), k_valid, np.nan)

    # Scatter back into full (b*m*n, s) / (b*m*n,) arrays (padded slots stay 0/nan).
    log_w_psis_np = np.zeros((b * m * n, s), dtype=log_w_psis_valid.dtype)
    log_w_psis_np[valid] = log_w_psis_valid
    k_np = np.full(b * m * n, np.nan, dtype=k_valid.dtype)
    k_np[valid] = k_valid

    log_w_psis = log_p.new_tensor(log_w_psis_np).reshape(b, m, n, s)
    k = log_p.new_tensor(k_np).reshape(b, m, n)

    # az.psislw returns log-normalized weights (logsumexp = 0), so:
    # log p_loo(y_ij) = logsumexp(log_w_psis_norm + log_p_ij, dim=samples)
    log_p_loo = torch.logsumexp(log_w_psis + log_p, dim=-1)   # (b, m, n)

    mask_f = mask.float()
    loo_nll = -(log_p_loo * mask_f).sum(dim=(1, 2)) / n_obs   # (b,)
    # nanmean over valid positions (nan = PSIS failed; inf k already replaced above)
    k[~mask] = float('nan')
    pareto_k = k.reshape(b, m * n).nanmean(dim=-1)     # (b,)

    return loo_nll, pareto_k


def posteriorPredictiveSample(
    pp: D.Distribution,
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


def posteriorPredictiveR2(
    pp: D.Distribution,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Posterior predictive R² for continuous outcomes. Returns (b,)."""
    y_obs = data['y']                        # (b, m, n)
    mask = data['mask_n'].float()            # (b, m, n)
    n = mask.sum(dim=(1, 2)).clamp(min=1)    # (b,)

    # posterior mean prediction: E[y | theta] averaged over samples
    yhat = pp.mean                           # (b, m, n, s)
    if w is not None:
        w_ = w.unsqueeze(1).unsqueeze(1)     # (b, 1, 1, s)
        yhat = (yhat * w_).sum(-1)
    else:
        yhat = yhat.mean(-1)                 # (b, m, n)
    yhat = yhat * mask

    # residual variance
    resid = (y_obs - yhat) * mask
    ss_res = resid.square().sum(dim=(1, 2))

    # total variance
    y_mean = (y_obs * mask).sum(dim=(1, 2)) / n
    ss_tot = ((y_obs - y_mean.unsqueeze(-1).unsqueeze(-1)) * mask).square().sum(dim=(1, 2))
    return 1 - ss_res / ss_tot.clamp(min=1e-12)


def posteriorPredictiveAUC(
    pp: D.Distribution,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Posterior predictive AUC for binary outcomes. Returns (b,)."""
    y_obs = data['y']                        # (b, m, n)
    mask = data['mask_n']                    # (b, m, n)

    # posterior mean predicted probability
    p = pp.probs                             # (b, m, n, s)
    if w is not None:
        w_ = w.unsqueeze(1).unsqueeze(1)
        p_mean = (p * w_).sum(-1)
    else:
        p_mean = p.mean(-1)                  # (b, m, n)

    # per-dataset AUC
    b = y_obs.shape[0]
    auc = y_obs.new_zeros(b)
    for i in range(b):
        m = mask[i].bool()
        yi = y_obs[i][m]
        pi = p_mean[i][m]
        n_pos = yi.sum()
        n_neg = yi.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            auc[i] = float('nan')
            continue
        # Wilcoxon-Mann-Whitney: fraction of (pos, neg) pairs correctly ranked
        pos_scores = pi[yi == 1]
        neg_scores = pi[yi == 0]
        auc[i] = (pos_scores.unsqueeze(-1) > neg_scores.unsqueeze(0)).float().mean()
    return auc


def posteriorPredictiveDeviance(
    pp: D.Distribution,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Posterior predictive mean deviance for count outcomes. Returns (b,)."""
    y_obs = data['y']                        # (b, m, n)
    mask = data['mask_n'].float()            # (b, m, n)
    n = mask.sum(dim=(1, 2)).clamp(min=1)    # (b,)

    # posterior mean predicted rate
    mu = pp.rate                             # (b, m, n, s)
    if w is not None:
        w_ = w.unsqueeze(1).unsqueeze(1)
        mu_mean = (mu * w_).sum(-1)
    else:
        mu_mean = mu.mean(-1)               # (b, m, n)
    mu_mean = mu_mean.clamp(min=1e-8)

    # deviance: 2 * sum[ y*log(y/mu) - (y - mu) ]
    y = y_obs.clamp(min=1e-8)
    dev = 2 * (y * torch.log(y / mu_mean) - (y_obs - mu_mean))
    return (dev * mask).sum(dim=(1, 2)) / n
