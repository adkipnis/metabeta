"""
posthoc/evidence.py — log p(y) by importance sampling (IS) from a Student-t importance function.

The flow-pool IS evidence (posthoc/importance.py) errs where the flow under-covers the
posterior; a multivariate t fitted per dataset to the refined draws covers it with heavier tails
(Geweke 1989, Econometrica 57:1317). The weights live on the standardized data, so batch and
draws must be unrescaled; the result is moved to the raw response scale, where a log Bayes factor
between two models of the same data is the difference of their log-evidences.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import distributions as D

from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.utils.dataloader import subsetBatch
from metabeta.utils.families import logProbCorrRfx
from metabeta.utils.preprocessing import logJacobianStandardization
from metabeta.utils.psis import psislw
from metabeta.utils.regularization import corrToLower, corrToUnconstrained, unconstrainedToCholesky
from metabeta.utils.results import Proposal

T_NU = 5  # dof of the t proposal
T_N_INNER = 4  # IS² draws per group (GLMMs)


def isSummary(lw: torch.Tensor) -> dict[str, np.ndarray]:
    """IS log-evidence, PSIS k and efficiency per row of log weights (b, S) on the cpu."""
    lw_psis, k = psislw(lw.numpy())
    w = torch.softmax(torch.as_tensor(lw_psis), -1)
    return {
        'log_ev': (torch.logsumexp(lw, -1) - math.log(lw.shape[-1])).numpy(),
        'k': np.asarray(k, dtype=float),
        'eff': (1.0 / (lw.shape[-1] * (w**2).sum(-1))).numpy(),
    }


def tEvidence(
    batch: dict[str, torch.Tensor],
    posterior: Proposal,
    lf: int,
    n_draws: int,
    gen: torch.Generator | None = None,
) -> dict[str, np.ndarray]:
    """log p(y) on the raw response scale, PSIS k and IS efficiency per dataset, (b,) each, from
    n_draws of a multivariate t (T_NU dof) fitted to each dataset's posterior draws in
    u = (β, log σ_rfx[, log σ_eps][, z_corr]); runs in float64 on the batch's device. Datasets
    are processed in groups sharing d, q and the LKJ switch, which fix the t's dimension. GLMMs
    use IS² weights (T_N_INNER draws per group), their Laplace mode search seeded at the
    per-group median of the posterior rfx draws.
    """
    batch = {
        k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in batch.items()
    }
    q = batch['mask_q'].sum(-1)
    structure = torch.stack([batch['mask_d'].sum(-1), q, (batch['eta_rfx'] > 0) & (q >= 2)], -1)
    out = {k: np.empty(len(q)) for k in ('log_ev', 'k', 'eff')}
    for key in structure.unique(dim=0):
        sel = (structure == key).all(-1)
        sel_np = sel.cpu().numpy()
        block = batch if sel_np.all() else subsetBatch(batch, sel_np)
        res = _tEvidenceBlock(block, posterior, sel, lf, n_draws, gen)
        for k in out:
            out[k][sel_np] = res[k]
    if posterior.has_sigma_eps:
        out['log_ev'] += logJacobianStandardization(batch).cpu().numpy()
    return out


def _tEvidenceBlock(
    batch64: dict[str, torch.Tensor],
    posterior: Proposal,
    sel: torch.Tensor,
    lf: int,
    n_draws: int,
    gen: torch.Generator | None,
) -> dict[str, np.ndarray]:
    """tEvidence on standardized data for the datasets `sel` of `posterior`, which share d, q and
    the LKJ switch; batch64 holds those datasets only."""
    dt, dev = torch.float64, batch64['X'].device
    b, m = batch64['X'].shape[:2]
    d_max, q_max = batch64['X'].shape[-1], batch64['Z'].shape[-1]
    d_i, q_i = int(batch64['mask_d'][0].sum()), int(batch64['mask_q'][0].sum())
    eta = batch64['eta_rfx']  # (b,)
    corr = bool(eta[0] > 0) and q_i >= 2
    eps = int(posterior.has_sigma_eps)
    dc_i, dc_max = (q_i * (q_i - 1) // 2 if corr else 0), posterior.d_corr
    sl_sig = slice(d_i, d_i + q_i + eps)

    # posterior draws in u, (b, s, p)
    parts = [
        posterior.ffx[sel][..., :d_i].to(dt),
        posterior.sigma_rfx[sel][..., :q_i].to(dt).clamp_min(1e-8).log(),
    ]
    if eps:
        parts.append(posterior.sigma_eps[sel].to(dt).clamp_min(1e-8).log().unsqueeze(-1))
    if corr:
        parts.append(corrToUnconstrained(posterior.corr_rfx[sel][..., :q_i, :q_i].to(dt)))
    u_post = torch.cat(parts, -1)
    p = u_post.shape[-1]

    # one t per dataset and n_draws fresh draws from it, (b, S, p)
    mu = u_post.mean(1, keepdim=True)  # (b, 1, p)
    c = u_post - mu
    cov = c.mT @ c / (u_post.shape[1] - 1) + 1e-6 * torch.eye(p, dtype=dt, device=dev)
    L = torch.linalg.cholesky(cov)  # (b, p, p)
    z = torch.randn(b, n_draws, p, generator=gen, dtype=dt, device=dev)
    chi2 = torch.randn(b, n_draws, T_NU, generator=gen, dtype=dt, device=dev).square().sum(-1)
    u = mu + (z @ L.mT) * (T_NU / chi2).sqrt().unsqueeze(-1)
    maha = torch.linalg.solve_triangular(L, (u - mu).mT, upper=False).square().sum(-2)  # (b, S)
    log_t = (
        math.lgamma((T_NU + p) / 2)
        - math.lgamma(T_NU / 2)
        - p / 2 * math.log(T_NU * math.pi)
        - L.diagonal(dim1=-2, dim2=-1).log().sum(-1, keepdim=True)
        - (T_NU + p) / 2 * torch.log1p(maha / T_NU)
    )

    # the draws in the batch's padded constrained layout; the far t tails reach correlations
    # that round to ±1, outside the LKJ support: density 0 there
    g = u.new_zeros(b, n_draws, d_max + q_max + eps + dc_max)
    g[..., :d_i] = u[..., :d_i]
    sigmas = u[..., sl_sig].exp()
    g[..., d_max : d_max + q_i] = sigmas[..., :q_i]
    if eps:
        g[..., d_max + q_max] = sigmas[..., -1]
    supported = torch.ones(b, n_draws, dtype=torch.bool, device=dev)
    if corr:
        z_corr = u[..., sl_sig.stop :]
        supported = D.constraints.corr_cholesky.check(unconstrainedToCholesky(z_corr, q_i))
        z_corr = torch.where(supported.unsqueeze(-1), z_corr, 0.0)
        L_corr = unconstrainedToCholesky(z_corr, q_i)
        off = d_max + q_max + eps
        g[..., off : off + dc_i] = corrToLower(L_corr @ L_corr.mT)
    if lf == 0:  # the marginal target ignores the rfx
        rfx = g.new_zeros(b, m, n_draws, q_max)
    else:  # Laplace mode-search seed
        rfx = posterior.rfx[sel].to(dt).median(dim=2, keepdim=True).values
        rfx = rfx.expand(b, m, n_draws, q_max)
    proposal = Proposal(
        {
            'global': {'samples': g, 'log_prob': g.new_zeros(b, n_draws)},
            'local': {'samples': rfx, 'log_prob': g.new_zeros(b, m, n_draws)},
        },
        has_sigma_eps=bool(eps),
        d_corr=dc_max,
    )

    # log p(y, θ) in u: exact marginal (Normal) or IS² (GLMMs), |dσ/d log σ|, LKJ over z
    if lf == 0:
        sampler = ImportanceSampler(batch64, marginal=True, corr_prior=False)
    else:
        sampler = LaplaceImportanceSampler(
            batch64, n_inner=T_N_INNER, corr_prior=False, likelihood_family=lf
        )
    ll, lp = sampler.unnormalizedPosterior(proposal)
    log_joint = ll + lp + u[..., sl_sig].sum(-1)
    if corr:
        z_pad = u.new_zeros(b, n_draws, dc_max)
        z_pad[..., :dc_i] = z_corr
        q_active = torch.full((b,), q_i, dtype=torch.long, device=dev)
        log_joint = log_joint + logProbCorrRfx(z_pad, q_max, eta, q_active=q_active)
    log_joint = torch.nan_to_num(log_joint, nan=-math.inf, posinf=-math.inf)
    return isSummary((torch.where(supported, log_joint, -math.inf) - log_t).cpu())
