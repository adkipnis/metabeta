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
    mode: Literal['expected', 'mixture', 'loo_proxy'] = 'mixture',
    log_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns per-dataset NLL with shape (b,).
    mode='expected':
        E_p(theta|D)[ -log p(y | theta) ]
    mode='mixture':
        -log E_p(theta|D)[ p(y | theta) ]
    mode='loo_proxy':
        log E_p(theta|D)[ 1/p(y | theta) ]  — naive IS LOO estimate; always >= mixture NLL
    """
    if log_p is None:
        y_obs = data['y'].unsqueeze(-1)   # (b, m, n, 1)
        log_p = pp.log_prob(y_obs)        # (b, m, n, s)
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
    elif mode == 'loo_proxy':
        if w is None:
            s = log_p.shape[-1]
            log_loo = torch.logsumexp(-log_p, dim=-1) - math.log(s)
        else:
            log_w = w.log().unsqueeze(1).unsqueeze(1)
            log_loo = torch.logsumexp(-log_p + log_w, dim=-1)
        return (log_loo * obs_mask).sum(dim=(1, 2)) / n_obs
    else:
        raise ValueError(f'unknown mode: {mode}')


def psisLooNLL(
    pp: D.Distribution,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
    reff: float = 1.0,
    log_p: torch.Tensor | None = None,
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
    reff : relative effective sample size (ESS / n_draws). Use 1.0 for i.i.d.
        samples (flow / IS). For MCMC, pass mean(ESS) / n_draws to set the
        correct PSIS tail threshold.
    log_p : precomputed log_prob tensor (b, m, n, s). When provided, skips the
        pp.log_prob call (avoids recomputing when sharing with posteriorPredictiveNLL).
    """

    if log_p is None:
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
    # Only run PSIS on real (non-padded) observations; scatter results back afterward.
    valid = mask.reshape(b * m * n).numpy()  # (b*m*n,) bool
    log_w_flat = log_w.detach().reshape(b * m * n, s).numpy()
    log_w_valid = log_w_flat[valid]          # (n_valid, s)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        log_w_psis_valid, k_valid = az.psislw(log_w_valid, reff=reff)

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


def _predQuantile(
    y_rep: torch.Tensor,
    levels: torch.Tensor,
    w: torch.Tensor | None,
) -> torch.Tensor:
    """Predictive quantiles at given probability levels.

    y_rep  : (b, m, n, s) — one draw per posterior sample
    levels : (n_levels,) in [0, 1], sorted ascending
    w      : (b, s) IS weights or None (uniform)

    Returns (n_levels, b, m, n).
    """
    b, m, n, s = y_rep.shape
    if w is None:
        return torch.quantile(y_rep, levels, dim=-1)   # (n_levels, b, m, n)

    w_norm = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
    w_exp = w_norm[:, None, None, :].expand(b, m, n, s)
    y_sorted, sort_idx = torch.sort(y_rep, dim=-1)
    w_sorted = torch.gather(w_exp, dim=-1, index=sort_idx)
    cdf = w_sorted.cumsum(dim=-1).contiguous()          # (b, m, n, s)

    cdf_flat = cdf.reshape(b * m * n, s)
    levels_exp = levels.unsqueeze(0).expand(b * m * n, -1).contiguous()
    idx = torch.searchsorted(cdf_flat, levels_exp, right=False).clamp(0, s - 1)
    q_flat = y_sorted.reshape(b * m * n, s).gather(1, idx)   # (b*m*n, n_levels)
    return q_flat.reshape(b, m, n, -1).permute(3, 0, 1, 2)   # (n_levels, b, m, n)


def _ppcNormal(
    pp: D.Normal,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    mask_f: torch.Tensor,
    n_obs: torch.Tensor,
    alphas: list[float],
    w: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast analytic path for Normal predictive distributions.

    Avoids sampling and sorting by working directly with the Normal CDF.

    Coverage: for each observation, compute the mixture CDF at y_obs —
        F(y) = mean_s[ Φ((y - mu_s) / sigma_s) ]
    then y is inside the (1-alpha) interval iff alpha/2 <= F(y) <= 1 - alpha/2.

    Width: moment approximation for the Normal mixture —
        std_mix = sqrt( E_s[sigma_s^2] + Var_s[mu_s] )
    which is exact when all components are identical and a tight lower bound
    otherwise. Sufficient for comparing dispersion across methods.
    """
    # mixture CDF at y_obs — shape (b, m, n)
    y_exp = y_obs.unsqueeze(-1)   # (b, m, n, 1) broadcasts against (b, m, n, s)
    if w is None:
        F_y = pp.cdf(y_exp).mean(dim=-1)
        mu_var = pp.mean.var(dim=-1)
        sigma2_mean = pp.variance.mean(dim=-1)
    else:
        w_norm = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
        w_exp = w_norm[:, None, None, :]
        F_y = (pp.cdf(y_exp) * w_exp).sum(dim=-1)
        mu = pp.mean
        mu_w = (mu * w_exp).sum(-1)
        mu_var = ((mu**2) * w_exp).sum(-1) - mu_w**2
        sigma2_mean = (pp.variance * w_exp).sum(-1)

    std_mix = (mu_var.clamp(min=0) + sigma2_mean).sqrt()   # (b, m, n)

    n_alphas = len(alphas)
    b = y_obs.shape[0]
    coverage = y_obs.new_zeros(n_alphas, b)
    width = y_obs.new_zeros(n_alphas, b)
    _std_normal = D.Normal(y_obs.new_zeros(()), y_obs.new_ones(()))

    for a, alpha in enumerate(alphas):
        inside = ((F_y >= alpha / 2) & (F_y <= 1 - alpha / 2)) & mask
        coverage[a] = inside.float().sum(dim=(1, 2)) / n_obs
        z = _std_normal.icdf(y_obs.new_tensor(1 - alpha / 2))
        width[a] = (2 * z * std_mix * mask_f).sum(dim=(1, 2)) / n_obs

    return coverage, width


def posteriorPredictiveCoverage(
    pp: D.Distribution,
    data: dict[str, torch.Tensor],
    w: torch.Tensor | None = None,
    alphas: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-sample predictive interval coverage and width for observed y.

    For Normal predictive distributions uses an analytic CDF path (fast,
    no sampling or sorting). For other families falls back to drawing one
    replicate per posterior sample and computing empirical quantiles.

    Comparing coverage and width between MB and NUTS reveals whether a
    LOO-NLL gap stems from over/under-dispersion in the predictive
    distribution rather than from posterior parameter quality
    (which ECE/NRMSE already capture).

    Parameters
    ----------
    w      : IS weights (b, s). When provided uses weighted statistics.
    alphas : significance levels. Defaults to the standard parameter ALPHAS.

    Returns
    -------
    coverage : (n_alphas, b) — fraction of y_obs inside each interval
    width    : (n_alphas, b) — mean predictive interval width over valid obs
    """
    from metabeta.evaluation.intervals import ALPHAS as _ALPHAS

    if alphas is None:
        alphas = _ALPHAS

    y_obs = data['y']       # (b, m, n)
    mask = data['mask_n']   # (b, m, n)
    mask_f = mask.float()
    n_obs = mask_f.sum(dim=(1, 2)).clamp(min=1)   # (b,)

    if isinstance(pp, D.Normal):
        with torch.no_grad():
            return _ppcNormal(pp, y_obs, mask, mask_f, n_obs, alphas, w)

    # Sampling fallback for non-Normal families
    with torch.no_grad():
        y_rep = pp.sample()   # (b, m, n, s)

    all_levels = sorted({q for alpha in alphas for q in (alpha / 2, 1 - alpha / 2)})
    levels_t = y_rep.new_tensor(all_levels)
    quantiles = _predQuantile(y_rep, levels_t, w)   # (n_levels, b, m, n)
    level_idx = {lv: i for i, lv in enumerate(all_levels)}

    n_alphas = len(alphas)
    b = y_obs.shape[0]
    coverage = y_obs.new_zeros(n_alphas, b)
    width = y_obs.new_zeros(n_alphas, b)

    for a, alpha in enumerate(alphas):
        q_lo = quantiles[level_idx[alpha / 2]]
        q_hi = quantiles[level_idx[1 - alpha / 2]]
        inside = ((y_obs >= q_lo) & (y_obs <= q_hi)) & mask
        coverage[a] = inside.float().sum(dim=(1, 2)) / n_obs
        width[a] = ((q_hi - q_lo) * mask_f).sum(dim=(1, 2)) / n_obs

    return coverage, width


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
