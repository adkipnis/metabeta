from typing import TYPE_CHECKING, Literal
import math
import numpy as np
import torch
from torch import distributions as D

from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import (
    hasSigmaEps,
    posteriorPredictiveDist,
    sampleFfxTorch,
    sampleSigmaTorch,
)

if TYPE_CHECKING:
    from metabeta.models.approximator import Approximator


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
    shape = (int(data['m'].max()),)
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


# ---------------------------------------------------------------------------
# Out-of-sample NLL
# ---------------------------------------------------------------------------


def makeOOSSplits(
    mask_n: torch.Tensor,
    n_splits: int,
    p_test: float = 0.2,
    generator: torch.Generator | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate n_splits independent random train/test observation splits.

    Each valid observation (where mask_n is True) is independently assigned to
    the test set with probability p_test. Groups that would otherwise have zero
    train observations have their first valid observation forced back to train.

    Args:
        mask_n: bool tensor of shape (B, m, n) marking valid observations.
        n_splits: number of independent splits to generate.
        p_test: probability of assigning an observation to the test set.
        generator: optional RNG for reproducibility.

    Returns:
        List of (train_mask, test_mask) pairs, each of shape (B, m, n).
    """
    splits = []
    for _ in range(n_splits):
        rand = torch.rand(mask_n.shape, generator=generator, device=mask_n.device)
        test_mask = (rand < p_test) & mask_n
        train_mask = mask_n & ~test_mask

        # Guarantee at least one train observation per group that has any data.
        # cumsum == 1 picks out the first valid position along the n dimension.
        first_valid = (mask_n.cumsum(dim=-1) == 1) & mask_n  # (B, m, n)
        starved = (train_mask.sum(dim=-1) == 0) & mask_n.any(dim=-1)  # (B, m)
        force_train = first_valid & starved.unsqueeze(-1)
        train_mask = train_mask | force_train
        test_mask = test_mask & ~force_train

        splits.append((train_mask, test_mask))
    return splits


def applyObsMask(
    data: dict[str, torch.Tensor],
    obs_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return a shallow copy of data with the observation mask and derived fields updated.

    Updates mask_n, ns, n, and mask_m consistently.  Does not copy or zero y, X, or Z:
    the model multiplies these by mask_n internally, so excluded positions are already
    treated as zero in all downstream computations.

    Args:
        data: batched dataset dict.
        obs_mask: bool tensor of shape (B, m, n) — the new mask_n to apply.

    Returns:
        Shallow copy of data with mask_n, ns, n, mask_m replaced.
    """
    ns = obs_mask.sum(dim=-1).to(data['ns'].dtype)   # (B, m)
    n = ns.sum(dim=-1).to(data['n'].dtype)            # (B,)
    mask_m = ns > 0                                    # (B, m)
    return {**data, 'mask_n': obs_mask, 'ns': ns, 'n': n, 'mask_m': mask_m}


def estimateOOSProposals(
    approximator: 'Approximator',
    data: dict[str, torch.Tensor],
    train_masks: list[torch.Tensor],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> list[Proposal]:
    """Estimate one posterior proposal per train split, processing all B datasets in parallel.

    For each train mask the approximator sees only the masked observations; the full
    batch of B datasets is processed in a single forward pass per split.

    Args:
        approximator: trained Approximator on the target device.
        data: full batched dataset on the same device as the approximator.
        train_masks: list of (B, m, n) bool tensors, one per split.
        n_samples: number of posterior samples to draw per dataset.
        rng: numpy Generator or None. Its state advances across splits, giving
            distinct samples per split from a single object.

    Returns:
        List of Proposal objects, one per split.
    """
    proposals = []
    for train_mask in train_masks:
        data_train = applyObsMask(data, train_mask)
        proposals.append(approximator.estimate(data_train, n_samples=n_samples, rng=rng))
    return proposals


def oosNLL(
    approximator: 'Approximator',
    data: dict[str, torch.Tensor],
    n_splits: int = 5,
    p_test: float = 0.2,
    n_samples: int = 500,
    likelihood_family: int = 0,
    mode: Literal['expected', 'mixture'] = 'mixture',
    generator: torch.Generator | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Estimate out-of-sample posterior predictive NLL via random within-group splits.

    For each split: condition the posterior on train observations only, then evaluate
    the predictive NLL on the held-out test observations. The posterior predictive
    distribution p(y_test | X, Z, theta) is evaluated at all positions (using the full
    X and Z), with the test mask applied only at the NLL scoring step.

    With mode='mixture' this approximates the ELPD:
        E[ log int p(y_test | theta) p(theta | y_train) dtheta ]

    Args:
        approximator: trained Approximator on the target device.
        data: batched dataset dict on the same device.
        n_splits: number of independent random splits; NLL is averaged over splits.
        p_test: fraction of observations held out per group per split.
        n_samples: posterior samples per split (trades off variance vs. cost).
        likelihood_family: 0=Normal, 1=Bernoulli, 2=Poisson.
        mode: 'mixture' for ELPD (recommended), 'expected' for E[-log p].
        generator: torch.Generator controlling the train/test split masks.
        rng: numpy Generator controlling the flow's base distribution sampler.
            Pass a freshly-seeded Generator on each call for reproducibility.

    Returns:
        Tensor of shape (B,) with per-dataset OOS NLL averaged over splits.
    """
    splits = makeOOSSplits(data['mask_n'], n_splits, p_test=p_test, generator=generator)
    train_masks = [train for train, _ in splits]
    test_masks = [test for _, test in splits]

    proposals = estimateOOSProposals(approximator, data, train_masks, n_samples, rng=rng)

    nlls = []
    for proposal, test_mask in zip(proposals, test_masks):
        # Evaluate p(y | X, Z, theta) at all positions using full X and Z.
        pp = getPosteriorPredictive(proposal, data, likelihood_family)
        # Score only the held-out test observations.
        data_test = applyObsMask(data, test_mask)
        nll = posteriorPredictiveNLL(pp, data_test, w=proposal.weights, mode=mode)
        nlls.append(nll)

    return torch.stack(nlls).mean(dim=0)  # (B,)
