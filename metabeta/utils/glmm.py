"""GLMM variance-component estimators for Bernoulli and Poisson likelihoods.

Uses PQL (Penalized Quasi-Likelihood) linearization in three stages:

Stage 0  Fixed-effects IRLS → overdispersion / scale-0 random-effects estimate
Stage 1  One PQL pass → Ψ̂_PQL and β̂_WG via weighted group-level Schur complement
Stage 2  Per-group Laplace Newton → Ψ̂_Lap, β̂_GLS, BLUPs

Supports batched inputs (B, m, n, *) with variable group and observation counts
via mask_n and mask_m padding.
"""

import torch
import torch.nn.functional as F

from metabeta.utils.least_squares import (
    _adaptive_ridge,
    irlsBernoulliCompacted,
    irlsPoissonCompacted,
)
from metabeta.utils.gls import _adaptive_ridge_bm


def _group_nll(
    bg: torch.Tensor,
    beta_0: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    Psi_inv: torch.Tensor,
    likelihood_family: int,
) -> torch.Tensor:
    """Per-group negative log-posterior (B, m) at random-effects bg.

    f(b_g) = -Σ_i log p(y_i | eta_i) + (1/2) b_g^T Ψ^{-1} b_g
    """
    eta = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    if likelihood_family == 1:
        nll_obs = (
            -ym * F.logsigmoid(eta)
            - (1.0 - ym) * F.logsigmoid(-eta)
        ) * mask_n
    else:
        mu = torch.exp(eta.clamp(max=20))
        nll_obs = (-(ym * eta - mu)) * mask_n
    nll_g = nll_obs.sum(dim=-1)  # (B, m)
    prior_g = 0.5 * torch.einsum('bmq,bqr,bmr->bm', bg, Psi_inv, bg)
    return nll_g + prior_g


def _psd_project(M: torch.Tensor) -> torch.Tensor:
    """Project a symmetric (B, q, q) matrix onto the PSD cone."""
    M = 0.5 * (M + M.mT)
    vals, vecs = torch.linalg.eigh(M)
    vals = vals.clamp(min=0.0)
    return vecs @ torch.diag_embed(vals) @ vecs.mT


def _pseudo_inverse(M: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse of a PSD (B, q, q) matrix via eigendecomposition."""
    vals, vecs = torch.linalg.eigh(M)
    tol = vals.amax(dim=-1, keepdim=True).clamp(min=1e-8) * 1e-6
    inv_vals = torch.where(vals > tol, 1.0 / vals.clamp(min=1e-30), torch.zeros_like(vals))
    return vecs @ torch.diag_embed(inv_vals) @ vecs.mT


def _pql_working(
    eta: torch.Tensor,
    y: torch.Tensor,
    mask_n: torch.Tensor,
    likelihood_family: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PQL working weights w and working response ytilde.

    Parameters
    ----------
    eta      : (B, m, n) linear predictor
    y        : (B, m, n) observed responses
    mask_n   : (B, m, n) 1 for active observations
    likelihood_family : 1 = Bernoulli/logit, 2 = Poisson/log

    Returns
    -------
    w        : (B, m, n) IRLS weights (0 for inactive obs)
    ytilde   : (B, m, n) working response
    """
    if likelihood_family == 1:
        mu = torch.sigmoid(eta)
        w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n
        ytilde = (eta + (y - mu) / w.clamp(min=1e-30)) * mask_n
    elif likelihood_family == 2:
        mu = torch.exp(eta.clamp(max=20))
        w = mu.clamp(min=1e-6) * mask_n
        ytilde = (eta + (y - mu) / w.clamp(min=1e-30)) * mask_n
    else:
        raise ValueError(f'unsupported likelihood_family={likelihood_family}')
    return w, ytilde


def glmmFull(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,  # (B, m)     1 for active groups
    ns: torch.Tensor,  # (B, m)     group sizes (float)
    n_total: torch.Tensor,  # (B,)       total active observations
    likelihood_family: int = 1,
    n_newton: int = 5,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM variance-component estimator.

    Parameters
    ----------
    Xm, ym, Zm  : batched grouped data tensors
    mask_n      : observation-level binary mask (B, m, n)
    mask_m      : group-level binary mask (B, m)
    ns          : per-group observation counts (B, m)
    n_total     : per-dataset total observation count (B,)
    likelihood_family : 1=Bernoulli, 2=Poisson
    n_newton    : Newton steps for the per-group Laplace mode

    Returns
    -------
    dict with keys: beta_est, sigma_rfx_est, blup_est,
                    phi_pearson, psi_0, Psi_pql, Psi_lap, mean_Hg_inv
    """
    B, m, n, d = Xm.shape
    q = Zm.shape[-1]
    N = n_total.float()                                # (B,)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,)
    active = mask_m.bool()                             # (B, m)
    mask4 = mask_m[:, :, None, None]                  # (B, m, 1, 1)

    eye_q = torch.eye(q, device=Xm.device, dtype=Xm.dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)

    # ------------------------------------------------------------------
    # Stage 0: pooled IRLS → beta_0, phi_pearson, psi_0
    # ------------------------------------------------------------------
    if likelihood_family == 1:
        beta_0 = irlsBernoulliCompacted(Xm, ym, mask_n)
    else:
        beta_0 = irlsPoissonCompacted(Xm, ym, mask_n)

    eta_0 = torch.einsum('bmnd,bd->bmn', Xm, beta_0)  # (B, m, n)

    if likelihood_family == 1:
        mu_0 = torch.sigmoid(eta_0)
        pearson = (ym - mu_0).square() / (mu_0 * (1.0 - mu_0)).clamp(min=1e-6)
    else:
        mu_0 = torch.exp(eta_0.clamp(max=20))
        pearson = (ym - mu_0).square() / mu_0.clamp(min=1e-6)

    pearson = pearson * mask_n                          # zero inactive
    sum_pearson = pearson.sum(dim=(1, 2))               # (B,)
    phi_pearson = sum_pearson / (N - d).clamp(min=1.0)  # (B,)

    # grand mean of mu_0 across active observations
    mu_bar = (mu_0 * mask_n).sum(dim=(1, 2)) / N.clamp(min=1.0)  # (B,)

    if likelihood_family == 2:
        psi_0 = ((phi_pearson - 1.0) / mu_bar.clamp(min=1e-6)).clamp(min=0.0)
    else:
        n_bar = N / G
        rho_hat = ((phi_pearson - 1.0) / (n_bar - 1.0).clamp(min=1.0)).clamp(min=0.0)
        denom = (mu_bar * (1.0 - mu_bar)).clamp(min=1e-6)
        psi_0 = (rho_hat / denom).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Stage 1: one PQL pass → ZWZ, XWX, XWZ, XWy, ZWy, beta_wg, bhat_ols, Psi_pql
    # ------------------------------------------------------------------
    # start from beta_0, bg=0
    eta_1 = eta_0  # bg=0 so no Z-contribution
    w1, ytilde1 = _pql_working(eta_1, ym, mask_n, likelihood_family)

    ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w1, Zm)          # (B, m, q, q)
    ZWy = torch.einsum('bmnq,bmn->bmq', Zm, w1 * ytilde1)           # (B, m, q)
    XWZ = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w1, Zm)          # (B, m, d, q)
    XWX = torch.einsum('bmnd,bmn,bmnk->bmdk', Xm, w1, Xm)          # (B, m, d, d)
    XWy = torch.einsum('bmnd,bmn->bmd', Xm, w1 * ytilde1)           # (B, m, d)

    # for inactive groups replace ZWZ with identity so solves stay non-singular
    ZWZ_safe = torch.where(active[:, :, None, None], ZWZ, eye_q)    # (B, m, q, q)
    ZWZ_inv = torch.linalg.solve(
        ZWZ_safe + _adaptive_ridge_bm(ZWZ_safe), eye_q_bm
    )                                                                 # (B, m, q, q)

    # beta_wg via Schur complement: A_g = XWX - XWZ ZWZ_inv ZWX (summed over active groups)
    ZWX = XWZ.mT                                                     # (B, m, q, d)
    ZWZ_inv_ZWX = torch.einsum('bmqr,bmrd->bmqd', ZWZ_inv, ZWX)     # (B, m, q, d)
    A_g = XWX - torch.einsum('bmdq,bmqk->bmdk', XWZ, ZWZ_inv_ZWX)   # (B, m, d, d)
    rhs_g = XWy - torch.einsum('bmdq,bmq->bmd', XWZ, torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZWy))

    # sum over active groups
    sum_A = (A_g * mask4).sum(dim=1)                                  # (B, d, d)
    sum_rhs = (rhs_g * mask_m[:, :, None]).sum(dim=1)                 # (B, d)
    beta_wg = torch.linalg.solve(sum_A + _adaptive_ridge(sum_A), sum_rhs)  # (B, d)

    # group-level naive OLS estimates of bg
    # Key: use beta_0 (pooled IRLS) not beta_wg, because beta_wg does not
    # estimate the components of β that lie in Z's column space — the same
    # reasoning as glsNormalFull's Stage 2 comment.  beta_0 is a consistent
    # estimator of all fixed-effect components, so E[b̂_g] ≈ b_g.
    resid1 = (ytilde1 - torch.einsum('bmnd,bd->bmn', Xm, beta_0)) * mask_n  # (B, m, n)
    ZW_resid1 = torch.einsum('bmnq,bmn->bmq', Zm, w1 * resid1)       # (B, m, q)
    bhat_ols = torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZW_resid1)     # (B, m, q)
    bhat_ols = bhat_ols * mask_m[:, :, None]                          # zero inactive

    # Psi_pql via MoM: E[bhat bhat^T] = Psi + ZWZ_inv  (since phi=1 for PQL)
    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat_ols, bhat_ols)   # (B, m, q, q)
    # mean over active groups
    mean_bhat_outer = (bhat_outer * mask4).sum(dim=1) / G[:, None, None]  # (B, q, q)
    mean_ZWZ_inv = (ZWZ_inv * mask4).sum(dim=1) / G[:, None, None]        # (B, q, q)

    Psi_raw_pql = mean_bhat_outer - mean_ZWZ_inv                      # (B, q, q)
    Psi_pql = _psd_project(Psi_raw_pql)                               # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 2: per-group Laplace Newton → Psi_lap, beta_gls, BLUPs
    # ------------------------------------------------------------------
    Psi_inv = _pseudo_inverse(Psi_pql)                                # (B, q, q)

    # Damped Newton cold-started from zeros with Armijo backtracking.
    # Pure Newton from far-from-mode starting points can overshoot into regions
    # where w = mu(1-mu) ~ 0, making subsequent Hessians near-singular and
    # causing oscillating divergence.  Armijo backtracking guarantees monotone
    # decrease of the (globally convex) negative log-posterior at every step.
    bg = ym.new_zeros(B, m, q)                                        # (B, m, q)
    for _ in range(n_newton):
        eta_t = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
        w_t, ytilde_t = _pql_working(eta_t, ym, mask_n, likelihood_family)

        ZWZ_t = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_t, Zm)     # (B, m, q, q)
        # gradient of log-posterior: Z^T(y - mu)*mask_n - Psi_inv @ bg
        if likelihood_family == 1:
            mu_t = torch.sigmoid(eta_t)
        else:
            mu_t = torch.exp(eta_t.clamp(max=20))
        Zty_resid = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_t) * mask_n)  # (B, m, q)
        grad_g = Zty_resid - torch.einsum('bqr,bmr->bmq', Psi_inv, bg)        # (B, m, q)

        # Hessian: ZWZ_t + Psi_inv; for inactive groups use eye_q
        ZWZ_t_safe = torch.where(active[:, :, None, None], ZWZ_t, eye_q)
        Hg = ZWZ_t_safe + Psi_inv[:, None]                            # (B, m, q, q)
        delta = torch.linalg.solve(Hg + _adaptive_ridge_bm(Hg), grad_g)  # (B, m, q)

        # Armijo backtracking: monotone decrease of NLL per group.
        # slope = grad_g . delta > 0 (delta is an ascent direction for log-post).
        slope = (grad_g * delta).sum(dim=-1)                           # (B, m)
        f_old = _group_nll(bg, beta_0, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
        alpha = torch.ones(B, m, device=Xm.device, dtype=Xm.dtype)    # (B, m)
        for _ls in range(10):
            bg_trial = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]
            f_new = _group_nll(bg_trial, beta_0, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
            # sufficient decrease (c=0.1): accept if NLL decreased by ≥ c*alpha*slope
            accept = (f_new <= f_old - 0.1 * alpha * slope) | ~active
            if accept.all():
                break
            alpha = torch.where(accept, alpha, alpha * 0.5)

        bg = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]

    # final working quantities at (beta_0, Laplace-mode bg)
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    w_f, ytilde_f = _pql_working(eta_f, ym, mask_n, likelihood_family)

    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)         # (B, m, q, q)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q)

    Hg_final = ZWZ_f_safe + Psi_inv[:, None]                          # (B, m, q, q)
    Hg_inv = torch.linalg.solve(
        Hg_final + _adaptive_ridge_bm(Hg_final), eye_q_bm
    )                                                                   # (B, m, q, q)
    Hg_inv = Hg_inv * mask4                                            # zero inactive

    # Psi_lap via MoM including posterior uncertainty
    bg_outer = torch.einsum('bmq,bmr->bmqr', bg, bg)                  # (B, m, q, q)
    Psi_raw_lap = (bg_outer + Hg_inv).sum(dim=1) / G[:, None, None]   # (B, q, q)
    Psi_lap = _psd_project(Psi_raw_lap)                                # (B, q, q)

    mean_Hg_inv = (Hg_inv * mask4).sum(dim=1) / G[:, None, None]      # (B, q, q)

    # beta_gls via Woodbury with Psi_lap
    Psi_lap_inv = _pseudo_inverse(Psi_lap)                             # (B, q, q)

    # recompute XWX, XWZ, XWy, ZWy at final working quantities
    XWX_f = torch.einsum('bmnd,bmn,bmnk->bmdk', Xm, w_f, Xm)         # (B, m, d, d)
    XWZ_f = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w_f, Zm)         # (B, m, d, q)
    XWy_f = torch.einsum('bmnd,bmn->bmd', Xm, w_f * ytilde_f)         # (B, m, d)
    ZWy_f = torch.einsum('bmnq,bmn->bmq', Zm, w_f * ytilde_f)         # (B, m, q)

    Kg = ZWZ_f_safe + Psi_lap_inv[:, None]                             # (B, m, q, q)
    Kg_inv = torch.linalg.solve(Kg + _adaptive_ridge_bm(Kg), eye_q_bm)  # (B, m, q, q)
    Kg_inv = Kg_inv * mask4                                             # zero inactive

    ZWX_f = XWZ_f.mT                                                   # (B, m, q, d)
    Kg_inv_ZWX = torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)       # (B, m, q, d)
    A_gls_g = XWX_f - torch.einsum('bmdq,bmqk->bmdk', XWZ_f, Kg_inv_ZWX)  # (B, m, d, d)
    Kg_inv_ZWy = torch.einsum('bmqr,bmr->bmq', Kg_inv, ZWy_f)         # (B, m, q)
    rhs_gls_g = XWy_f - torch.einsum('bmdq,bmq->bmd', XWZ_f, Kg_inv_ZWy)  # (B, m, d)

    sum_A_gls = (A_gls_g * mask4).sum(dim=1)                           # (B, d, d)
    sum_rhs_gls = (rhs_gls_g * mask_m[:, :, None]).sum(dim=1)          # (B, d)
    beta_gls = torch.linalg.solve(
        sum_A_gls + _adaptive_ridge(sum_A_gls), sum_rhs_gls
    )                                                                   # (B, d)

    # BLUPs at beta_gls
    resid_gls = (ytilde_f - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    ZW_resid_gls = torch.einsum('bmnq,bmn->bmq', Zm, w_f * resid_gls)  # (B, m, q)
    blups = torch.einsum('bmqr,bmr->bmq', Kg_inv, ZW_resid_gls)        # (B, m, q)
    blups = blups * mask_m[:, :, None]

    # ------------------------------------------------------------------
    # cleanup and pack outputs
    # ------------------------------------------------------------------
    beta_gls = beta_gls.nan_to_num(nan=0.0)
    blups = blups.nan_to_num(nan=0.0)
    Psi_lap = Psi_lap.nan_to_num(nan=0.0, posinf=0.0)
    Psi_pql = Psi_pql.nan_to_num(nan=0.0, posinf=0.0)
    mean_Hg_inv = mean_Hg_inv.nan_to_num(nan=0.0, posinf=0.0)

    sigma_rfx_est = Psi_lap.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()  # (B, q)
    sigma_rfx_est = sigma_rfx_est.nan_to_num(nan=0.0, posinf=0.0)

    return {
        'beta_est': beta_gls,  # (B, d)
        'sigma_rfx_est': sigma_rfx_est,  # (B, q)
        'blup_est': blups,  # (B, m, q)
        'phi_pearson': phi_pearson,  # (B,)
        'psi_0': psi_0,  # (B,)
        'Psi_pql': Psi_pql,  # (B, q, q)
        'Psi_lap': Psi_lap,  # (B, q, q)
        'mean_Hg_inv': mean_Hg_inv,  # (B, q, q)
    }
