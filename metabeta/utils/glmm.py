"""GLMM variance-component estimators for Bernoulli and Poisson likelihoods.

Uses PQL (Penalized Quasi-Likelihood) linearization in three stages:

Stage 0  Pooled IRLS → β₀, overdispersion φ, scale-0 random-effects estimate ψ₀
Stage 1  One PQL pass at (β₀, b=0) → Ψ̂_PQL via Henderson MoM
Stage 2  Per-group damped Newton (Armijo backtracking) → Laplace mode b̂_g,
         then Ψ̂_Lap, β̂_GLS, BLUPs

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


def _pqlWorking(
    eta: torch.Tensor,
    y: torch.Tensor,
    mask_n: torch.Tensor,
    likelihood_family: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PQL working weights w and working response ỹ at linearisation point η.

    Bernoulli/logit : w = μ(1-μ),  ỹ = η + (y-μ)/w
    Poisson/log     : w = μ,        ỹ = η + (y-μ)/w
    Inactive observations are zeroed via mask_n.
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


def _psdProject(M: torch.Tensor) -> torch.Tensor:
    """Project a symmetric (B, q, q) matrix onto the PSD cone."""
    M = 0.5 * (M + M.mT)
    vals, vecs = torch.linalg.eigh(M)
    return vecs @ torch.diag_embed(vals.clamp(min=0.0)) @ vecs.mT


def _pseudoInverse(M: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse of a PSD (B, q, q) matrix via eigendecomposition."""
    vals, vecs = torch.linalg.eigh(M)
    tol = vals.amax(dim=-1, keepdim=True).clamp(min=1e-8) * 1e-6
    inv_vals = torch.where(vals > tol, 1.0 / vals.clamp(min=1e-30), torch.zeros_like(vals))
    return vecs @ torch.diag_embed(inv_vals) @ vecs.mT


def _groupNll(
    bg: torch.Tensor,
    beta_0: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    Psi_inv: torch.Tensor,
    likelihood_family: int,
) -> torch.Tensor:
    """Per-group negative log-posterior  f(bᵍ) = −Σᵢ log p(yᵢ|ηᵢ) + ½ bᵍᵀΨ⁻¹bᵍ.

    Returns shape (B, m).  Used by the Armijo line-search inside the Newton loop.
    """
    eta = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    if likelihood_family == 1:
        nll_obs = (-ym * F.logsigmoid(eta) - (1.0 - ym) * F.logsigmoid(-eta)) * mask_n
    else:
        mu = torch.exp(eta.clamp(max=20))
        nll_obs = (-(ym * eta - mu)) * mask_n
    return nll_obs.sum(dim=-1) + 0.5 * torch.einsum('bmq,bqr,bmr->bm', bg, Psi_inv, bg)


def glmmFull(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,  # (B, m)     1 for active groups
    ns: torch.Tensor,  # (B, m)     group sizes (float)
    n_total: torch.Tensor,  # (B,)       total active observations
    likelihood_family: int = 1,
    n_newton: int = 3,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM variance-component estimator.

    Parameters
    ----------
    Xm, ym, Zm  : batched grouped data tensors
    mask_n      : observation-level binary mask (B, m, n)
    mask_m      : group-level binary mask (B, m)
    ns          : per-group observation counts (B, m)
    n_total     : per-dataset total observation count (B,)
    likelihood_family : 1=Bernoulli/logit, 2=Poisson/log
    n_newton    : damped Newton steps for the per-group Laplace mode

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
    # Stage 0: pooled IRLS → β₀, overdispersion φ, scale-0 estimate ψ₀
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

    phi_pearson = (pearson * mask_n).sum(dim=(1, 2)) / (N - d).clamp(min=1.0)  # (B,)
    mu_bar = (mu_0 * mask_n).sum(dim=(1, 2)) / N.clamp(min=1.0)                # (B,)

    if likelihood_family == 2:
        psi_0 = ((phi_pearson - 1.0) / mu_bar.clamp(min=1e-6)).clamp(min=0.0)
    else:
        n_bar = N / G
        rho_hat = ((phi_pearson - 1.0) / (n_bar - 1.0).clamp(min=1.0)).clamp(min=0.0)
        psi_0 = (rho_hat / (mu_bar * (1.0 - mu_bar)).clamp(min=1e-6)).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Stage 1: one PQL pass at (β₀, b=0) → ZWZ⁻¹, b̂_OLS, Ψ̂_PQL
    # ------------------------------------------------------------------
    w1, _ = _pqlWorking(eta_0, ym, mask_n, likelihood_family)  # bg=0, so eta = eta_0

    ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w1, Zm)          # (B, m, q, q)

    # Replace inactive-group ZWZ with identity to keep solves non-singular
    ZWZ_safe = torch.where(active[:, :, None, None], ZWZ, eye_q)
    ZWZ_inv = torch.linalg.solve(ZWZ_safe + _adaptive_ridge_bm(ZWZ_safe), eye_q_bm)  # (B, m, q, q)

    # b̂_OLS = ZWZ⁻¹ Zᵀ(y−μ).  Use β₀ residuals (not β_WG) because β_WG omits
    # components in Z's column space when Z ⊂ span(X); β₀ estimates all d components.
    ZtYmMu = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_0) * mask_n)  # (B, m, q)
    bhat_ols = torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZtYmMu) * mask_m[:, :, None]

    # Ψ̂_PQL via Henderson MoM: E[b̂ b̂ᵀ] = Ψ + ZWZ⁻¹
    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat_ols, bhat_ols)       # (B, m, q, q)
    mean_bhat_outer = (bhat_outer * mask4).sum(dim=1) / G[:, None, None]
    mean_ZWZ_inv = (ZWZ_inv * mask4).sum(dim=1) / G[:, None, None]
    Psi_pql = _psdProject(mean_bhat_outer - mean_ZWZ_inv)                 # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 2: damped Newton → Laplace mode b̂_g, Ψ̂_Lap, β̂_GLS, BLUPs
    # ------------------------------------------------------------------
    Psi_inv = _pseudoInverse(Psi_pql)                                     # (B, q, q)

    # Warm-start from b̂_OLS.  Armijo backtracking guarantees monotone decrease
    # of the per-group negative log-posterior (globally convex) at every step,
    # so convergence is assured regardless of how far b̂_OLS is from the mode.
    bg = bhat_ols.clone()
    for _ in range(n_newton):
        eta_t = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)

        if likelihood_family == 1:
            mu_t = torch.sigmoid(eta_t)
        else:
            mu_t = torch.exp(eta_t.clamp(max=20))
        w_t = (mu_t * (1.0 - mu_t) if likelihood_family == 1 else mu_t).clamp(min=1e-6) * mask_n

        ZWZ_t = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_t, Zm)         # (B, m, q, q)
        grad_g = (
            torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_t) * mask_n)      # Zᵀ(y−μ)
            - torch.einsum('bqr,bmr->bmq', Psi_inv, bg)                   # −Ψ⁻¹b
        )                                                                   # (B, m, q)

        ZWZ_t_safe = torch.where(active[:, :, None, None], ZWZ_t, eye_q)
        Hg = ZWZ_t_safe + Psi_inv[:, None]                                # (B, m, q, q)
        delta = torch.linalg.solve(Hg + _adaptive_ridge_bm(Hg), grad_g)  # (B, m, q)

        # Armijo backtracking: halve step until sufficient decrease (c = 0.1).
        # slope = ∇logpost · δ > 0, so −slope < 0 is the NLL descent rate.
        slope = (grad_g * delta).sum(dim=-1)                               # (B, m)
        f_old = _groupNll(bg, beta_0, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
        alpha = torch.ones(B, m, device=Xm.device, dtype=Xm.dtype)
        for _ls in range(10):
            bg_trial = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]
            f_new = _groupNll(bg_trial, beta_0, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
            accept = (f_new <= f_old - 0.1 * alpha * slope) | ~active
            if accept.all():
                break
            alpha = torch.where(accept, alpha, alpha * 0.5)

        bg = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]

    # Final working quantities at (β₀, b̂_g)
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    w_f, ytilde_f = _pqlWorking(eta_f, ym, mask_n, likelihood_family)
    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)             # (B, m, q, q)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q)

    # Ψ̂_Lap = (1/G) Σ_g (b̂_g b̂_gᵀ + H_g⁻¹)  — MoM with posterior uncertainty
    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv = torch.linalg.solve(Hg_f + _adaptive_ridge_bm(Hg_f), eye_q_bm) * mask4
    bg_outer = torch.einsum('bmq,bmr->bmqr', bg, bg)
    Psi_lap = _psdProject((bg_outer + Hg_inv).sum(dim=1) / G[:, None, None])  # (B, q, q)
    mean_Hg_inv = (Hg_inv * mask4).sum(dim=1) / G[:, None, None]

    # β̂_GLS via Woodbury with Ψ̂_Lap
    Psi_lap_inv = _pseudoInverse(Psi_lap)
    XWX_f = torch.einsum('bmnd,bmn,bmnk->bmdk', Xm, w_f, Xm)             # (B, m, d, d)
    XWZ_f = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w_f, Zm)             # (B, m, d, q)
    XWy_f = torch.einsum('bmnd,bmn->bmd', Xm, w_f * ytilde_f)             # (B, m, d)
    ZWy_f = torch.einsum('bmnq,bmn->bmq', Zm, w_f * ytilde_f)             # (B, m, q)

    Kg = ZWZ_f_safe + Psi_lap_inv[:, None]                                 # (B, m, q, q)
    Kg_inv = torch.linalg.solve(Kg + _adaptive_ridge_bm(Kg), eye_q_bm) * mask4

    ZWX_f = XWZ_f.mT                                                       # (B, m, q, d)
    A_g = XWX_f - torch.einsum(                                            # Schur complement
        'bmdq,bmqk->bmdk', XWZ_f, torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    )
    rhs_g = XWy_f - torch.einsum('bmdq,bmq->bmd', XWZ_f,
                                  torch.einsum('bmqr,bmr->bmq', Kg_inv, ZWy_f))

    sum_A = (A_g * mask4).sum(dim=1)
    beta_gls = torch.linalg.solve(
        sum_A + _adaptive_ridge(sum_A),
        (rhs_g * mask_m[:, :, None]).sum(dim=1),
    )                                                                       # (B, d)

    # BLUPs: K_g⁻¹ Zᵀ W (ỹ − X β̂_GLS)
    resid_gls = (ytilde_f - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    blups = torch.einsum(
        'bmqr,bmr->bmq', Kg_inv,
        torch.einsum('bmnq,bmn->bmq', Zm, w_f * resid_gls),
    ) * mask_m[:, :, None]                                                 # (B, m, q)

    # ------------------------------------------------------------------
    # Pack outputs
    # ------------------------------------------------------------------
    beta_gls = beta_gls.nan_to_num(nan=0.0)
    blups = blups.nan_to_num(nan=0.0)
    Psi_lap = Psi_lap.nan_to_num(nan=0.0, posinf=0.0)
    Psi_pql = Psi_pql.nan_to_num(nan=0.0, posinf=0.0)
    mean_Hg_inv = mean_Hg_inv.nan_to_num(nan=0.0, posinf=0.0)
    sigma_rfx_est = Psi_lap.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    return {
        'beta_est': beta_gls,           # (B, d)
        'sigma_rfx_est': sigma_rfx_est, # (B, q)
        'blup_est': blups,              # (B, m, q)
        'phi_pearson': phi_pearson,     # (B,)
        'psi_0': psi_0,                 # (B,)
        'Psi_pql': Psi_pql,             # (B, q, q)
        'Psi_lap': Psi_lap,             # (B, q, q)
        'mean_Hg_inv': mean_Hg_inv,     # (B, q, q)
    }
