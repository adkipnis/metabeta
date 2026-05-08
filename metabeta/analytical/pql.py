"""PQL estimators for Bernoulli and Poisson GLMMs."""

import torch

from metabeta.analytical.glm import (
    _adaptiveRidge,
    _safeSolve,
    irlsBernoulliCompacted,
    irlsPoissonCompacted,
)
from metabeta.analytical.constants import (
    _BERNOULLI_BLUP_KH_VAR_CAP,
    _BERNOULLI_BLUP_VAR_INFLATION,
    _BERNOULLI_CORR_SHRINKAGE_C,
    _BERNOULLI_HG_INV_EIG_CAP,
    _BERNOULLI_HIGH_D_BLUP_VAR_FLOOR,
    _BERNOULLI_INITIAL_PSI_FLOOR,
    _BERNOULLI_PSI_EIG_CAP,
    _POISSON_BETA_CLAMP,
    _POISSON_BLUP_CLAMP,
    _POISSON_CORR_SHRINKAGE_C,
    _POISSON_HG_INV_EIG_CAP,
    _POISSON_INITIAL_PSI_FLOOR,
    _POISSON_PSI_EIG_CAP,
)
from metabeta.analytical.linalg import (
    _adaptiveRidgeBm,
    _eighWithJitter,
    _psdClampEigenvalues,
    _psdProject,
    _pseudoInverse,
    _ridgeInv,
    _shrinkOffDiagonal,
)
from metabeta.analytical.working import (
    _groupNll,
    _poissonMeanDerivative,
    _pqlWorking,
)


def _pqlPass(
    beta: torch.Tensor,  # (B, d) fixed effects for this pass
    Psi_inv: torch.Tensor,  # (B, q, q) precision used for Newton penalty and Hg
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)
    mask_m: torch.Tensor,  # (B, m)
    mask4: torch.Tensor,  # (B, m, 1, 1)
    active: torch.Tensor,  # (B, m) bool
    eye_q: torch.Tensor,  # (q, q)
    eye_q_bm: torch.Tensor,  # (B, m, q, q)
    G: torch.Tensor,  # (B,)
    likelihood_family: int,
    n_newton: int = 3,
    bg_init: torch.Tensor | None = None,  # warm start; None = cold start from zeros
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """One PQL pass: damped Newton → Ψ̂_Lap M-step → GLS β̂ and BLUPs.

    Newton starts from bg_init (or zeros if None) under (beta, Psi_inv).
    After Newton, computes Ψ̂_Lap = mean_g(b̂_g b̂_g' + H_g^{-1}), then
    β̂_GLS and BLUPs via the Woodbury/Schur-complement GLS under Ψ̂_Lap.

    bg is clamped to ±20 after each Newton step to prevent bg_outer from
    inflating Psi_lap for overdispersed Poisson or ill-conditioned datasets.

    Returns
    -------
    beta_gls   : (B, d)       GLS-corrected fixed effects
    Psi_lap    : (B, q, q)    Laplace M-step estimate of Ψ
    blups      : (B, m, q)    per-group random effects
    Kg_inv     : (B, m, q, q) GLS posterior covariance (ZWZ + Psi_lap_inv)^{-1}
    mean_Hg_inv: (B, q, q)    mean Laplace posterior covariance across groups
    resid_gls  : (B, m, n)    working residual ỹ − Xβ̂_GLS (masked)
    blup_kh_var: (B, m, q)    beta-estimation uncertainty contribution to blup_var
    """
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]

    # --- Newton loop ---
    # Cold start from zeros (Pass 1) or warm start from previous blups (Pass 2+).
    # bg is clamped at ±20 after each step: prevents bg_outer from inflating Psi_lap
    # for Poisson datasets with large counts, where 3 unconstrained Newton steps can
    # push bg to O(100) and Psi_lap to O(10000).
    bg = bg_init.clone() if bg_init is not None else ym.new_zeros(B, m, q)
    for _ in range(n_newton):
        eta_t = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
        if likelihood_family == 1:
            mu_t = torch.sigmoid(eta_t)
            score_t = ym - mu_t
            w_t = (mu_t * (1.0 - mu_t)).clamp(min=1e-6) * mask_n
        else:
            mu_t, deriv_t = _poissonMeanDerivative(eta_t)
            score_t = (ym - mu_t) * deriv_t
            w_t = mu_t * deriv_t.square() * mask_n
        ZWZ_t = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_t, Zm)         # (B, m, q, q)
        grad_g = torch.einsum('bmnq,bmn->bmq', Zm, score_t * mask_n) - torch.einsum(  # Zᵀscore
            'bqr,bmr->bmq', Psi_inv, bg
        )  # −Ψ⁻¹b
        ZWZ_t_safe = torch.where(active[:, :, None, None], ZWZ_t, eye_q)
        Hg = ZWZ_t_safe + Psi_inv[:, None]
        delta = _safeSolve(Hg + _adaptiveRidgeBm(Hg), grad_g)             # (B, m, q)
        slope = (grad_g * delta).sum(dim=-1)                               # (B, m)
        f_old = _groupNll(bg, beta, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
        alpha = torch.ones(B, m, device=Xm.device, dtype=Xm.dtype)
        for _ls in range(10):
            bg_trial = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]
            f_new = _groupNll(bg_trial, beta, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
            accept = (f_new <= f_old - 0.1 * alpha * slope) | ~active
            if accept.all():
                break
            alpha = torch.where(accept, alpha, alpha * 0.5)
        bg = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]
        bg = bg.clamp(-20.0, 20.0)  # prevent bg_outer blow-up in Psi_lap M-step

    # --- Final working quantities at (beta, bg) ---
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    w_f, ytilde_f = _pqlWorking(eta_f, ym, mask_n, likelihood_family)
    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)             # (B, m, q, q)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q)

    # Ψ̂_Lap M-step: Ψ = mean_g(b̂_g b̂_g' + H_g^{-1})
    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4

    if likelihood_family == 1:
        ns_g = mask_n.sum(dim=-1).clamp(min=1.0)                         # (B, m)
        y_rate = (ym * mask_n).sum(dim=-1) / ns_g                         # (B, m)
        outcome_balance = (4.0 * y_rate * (1.0 - y_rate)).clamp(0.0, 1.0)  # (B, m)
        info_g = ZWZ_f.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=0.0)
        info_weight = info_g / (info_g + float(q))
        cov_weight = (outcome_balance.sqrt() * info_weight).clamp(0.0, 1.0) * mask_m
        Hg_inv = _psdClampEigenvalues(Hg_inv, _BERNOULLI_HG_INV_EIG_CAP)
        Hg_inv = Hg_inv * cov_weight[:, :, None, None]
    elif likelihood_family == 2:
        info_g = ZWZ_f.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=0.0)
        info_weight = info_g / (info_g + float(q))
        Hg_inv = _psdClampEigenvalues(Hg_inv, _POISSON_HG_INV_EIG_CAP)
        Hg_inv = Hg_inv * (info_weight * mask_m)[:, :, None, None]

    bg_outer = torch.einsum('bmq,bmr->bmqr', bg, bg)
    Psi_lap = _psdProject((bg_outer + Hg_inv).sum(dim=1) / G[:, None, None])  # (B, q, q)
    if likelihood_family == 1:
        Psi_lap = _psdClampEigenvalues(Psi_lap, _BERNOULLI_PSI_EIG_CAP)
    elif likelihood_family == 2:
        Psi_lap = _psdClampEigenvalues(Psi_lap, _POISSON_PSI_EIG_CAP)
    mean_Hg_inv = (Hg_inv * mask4).sum(dim=1) / G[:, None, None]

    # --- β̂_GLS via Woodbury/Schur complement under freshly computed Ψ̂_Lap ---
    Psi_lap_inv = _pseudoInverse(Psi_lap)
    XWX_f = torch.einsum('bmnd,bmn,bmnk->bmdk', Xm, w_f, Xm)             # (B, m, d, d)
    XWZ_f = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w_f, Zm)             # (B, m, d, q)
    XWy_f = torch.einsum('bmnd,bmn->bmd', Xm, w_f * ytilde_f)             # (B, m, d)
    ZWy_f = torch.einsum('bmnq,bmn->bmq', Zm, w_f * ytilde_f)             # (B, m, q)

    Kg = ZWZ_f_safe + Psi_lap_inv[:, None]                                 # (B, m, q, q)
    Kg_inv = _safeSolve(Kg + _adaptiveRidgeBm(Kg), eye_q_bm) * mask4

    ZWX_f = XWZ_f.mT                                                       # (B, m, q, d)
    A_g = XWX_f - torch.einsum(  # Schur complement
        'bmdq,bmqk->bmdk', XWZ_f, torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    )
    rhs_g = XWy_f - torch.einsum(
        'bmdq,bmq->bmd', XWZ_f, torch.einsum('bmqr,bmr->bmq', Kg_inv, ZWy_f)
    )
    sum_A = (A_g * mask4).sum(dim=1)
    sum_A_reg = sum_A + _adaptiveRidge(sum_A)
    beta_gls = _safeSolve(
        sum_A_reg,
        (rhs_g * mask_m[:, :, None]).sum(dim=1),
    )                                                                       # (B, d)

    # --- BLUPs: K_g⁻¹ Zᵀ W (ỹ − X β̂_GLS) ---
    # Clamp beta_gls before residuals: near-singular GLS produces extreme values that
    # cause eta overflow in the next Newton pass or catastrophic BLUP outliers.
    beta_clamp = _POISSON_BETA_CLAMP if likelihood_family == 2 else 50.0
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-beta_clamp, beta_clamp)
    resid_gls = (ytilde_f - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    blups = (
        torch.einsum(
            'bmqr,bmr->bmq',
            Kg_inv,
            torch.einsum('bmnq,bmn->bmq', Zm, w_f * resid_gls),
        )
        * mask_m[:, :, None]
    )                                                 # (B, m, q)

    blup_clamp = _POISSON_BLUP_CLAMP if likelihood_family == 2 else 20.0
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-blup_clamp, blup_clamp)

    eye_d = torch.eye(d, device=Xm.device, dtype=Xm.dtype).expand(B, d, d)
    beta_var = _safeSolve(sum_A_reg, eye_d).diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
    K_ZWX = torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    blup_kh_var = (K_ZWX.square() * beta_var[:, None, None, :]).sum(dim=-1)
    blup_kh_var = (blup_kh_var * mask_m[:, :, None]).nan_to_num(nan=0.0, posinf=0.0)
    if likelihood_family == 1:
        kh_scale = max(min((d - 8) / 8.0, 1.0), 0.0)
        blup_kh_var = blup_kh_var.clamp(max=_BERNOULLI_BLUP_KH_VAR_CAP * kh_scale)
    else:
        blup_kh_var = torch.zeros_like(blup_kh_var)

    # Return Kg_inv for blup_var: (ZWZ + Psi_lap^{-1})^{-1} is the posterior covariance
    # of b_g under the final Psi_lap estimate — consistent with the Normal path's se²·W_g.
    return beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls, blup_kh_var


def _lmmGlmm(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,  # (B, m)     1 for active groups
    ns: torch.Tensor,  # (B, m)     group sizes (float)
    n_total: torch.Tensor,  # (B,)       total active observations
    likelihood_family: int,
    n_newton: int = 3,
    uncorr: torch.Tensor | None = None,  # (B,) bool — force Ψ diagonal for these datasets
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM variance-component estimator (private).

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
    dict with keys: beta_est, beta_wg, sigma_rfx_est, blup_est, blup_var,
                    bhat, resid_g, phi_pearson, psi_0, Psi_pql, Psi_lap, mean_Hg_inv
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
        mu_0, deriv_0 = _poissonMeanDerivative(eta_0)
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

    ZWZ_safe = torch.where(active[:, :, None, None], ZWZ, eye_q)
    ZWZ_inv = _safeSolve(ZWZ_safe + _adaptiveRidgeBm(ZWZ_safe), eye_q_bm)  # (B, m, q, q)

    if likelihood_family == 2:
        score_0 = (ym - mu_0) * deriv_0
    else:
        score_0 = ym - mu_0
    ZtYmMu = torch.einsum('bmnq,bmn->bmq', Zm, score_0 * mask_n)  # (B, m, q)
    bhat_ols = torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZtYmMu) * mask_m[:, :, None]

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat_ols, bhat_ols)       # (B, m, q, q)
    mean_bhat_outer = (bhat_outer * mask4).sum(dim=1) / G[:, None, None]
    mean_ZWZ_inv = (ZWZ_inv * mask4).sum(dim=1) / G[:, None, None]
    Psi_pql = _psdProject(mean_bhat_outer - mean_ZWZ_inv)                 # (B, q, q)

    # Floor eigenvalues at psi_0 (overdispersion-based RE scale estimate).
    # Prevents degenerate near-zero Ψ̂_PQL after PSD projection, which would
    # make Ψ̂_PQL⁻¹ → ∞ and kill Newton updates.
    vals_pql, vecs_pql = _eighWithJitter(Psi_pql)
    vals_pql = vals_pql.clamp(min=psi_0[:, None])
    Psi_pql = vecs_pql @ torch.diag_embed(vals_pql) @ vecs_pql.mT        # (B, q, q)
    if uncorr is not None:
        Psi_pql = torch.where(
            uncorr[:, None, None], torch.diag_embed(Psi_pql.diagonal(dim1=-2, dim2=-1)), Psi_pql
        )

    # ------------------------------------------------------------------
    # Stage 2: alternating Newton–GLS loop, up to max_passes=6.
    # Pass 1 cold-starts Newton under pooled β₀ and a stabilized Ψ̂_PQL inverse.
    # Passes 2–max_passes warm-start from previous BLUPs under β̂_GLS and
    # ridge-inv(Ψ̂_Lap); early exit when the 95th-percentile change in both
    # β and diag(Ψ̂) falls below 1e-3.
    # ------------------------------------------------------------------
    pass_args = (
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        mask4,
        active,
        eye_q,
        eye_q_bm,
        G,
        likelihood_family,
        n_newton,
    )
    if likelihood_family == 1:
        psi_0_floor = psi_0.clamp(min=_BERNOULLI_INITIAL_PSI_FLOOR)
    elif likelihood_family == 2:
        psi_0_floor = psi_0.clamp(min=_POISSON_INITIAL_PSI_FLOOR)
    else:
        psi_0_floor = psi_0.clamp(min=1e-6)
    max_passes = 6

    def _sanitize(b_gls: torch.Tensor, psi: torch.Tensor):
        """Clamp beta_gls and clean Psi_lap before using as next-pass inputs."""
        beta_clamp = _POISSON_BETA_CLAMP if likelihood_family == 2 else 50.0
        return (
            b_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-beta_clamp, beta_clamp),
            psi.nan_to_num(nan=0.0, posinf=0.0),
        )

    # Pass 1: pooled β₀, cold start.  Discrete outcomes can make Psi_pql exactly
    # zero in weakly identified directions; use a weak variance floor for shrinkage.
    Psi_inv = (
        _ridgeInv(Psi_pql, psi_0_floor) if likelihood_family in (1, 2) else _pseudoInverse(Psi_pql)
    )
    beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls, blup_kh_var = _pqlPass(
        beta_0, Psi_inv, *pass_args
    )
    beta_gls, Psi_lap = _sanitize(beta_gls, Psi_lap)
    if uncorr is not None:
        Psi_lap = torch.where(
            uncorr[:, None, None], torch.diag_embed(Psi_lap.diagonal(dim1=-2, dim2=-1)), Psi_lap
        )

    # Passes 2–max_passes: warm start, ridge-regularized Ψ̂_Lap, until convergence.
    # Each pass refines (β, b̂_g, Ψ̂_Lap) jointly. Early exit when the 95th-percentile
    # change across the batch is below tolerance — avoids running all max_passes when a
    # few pathological datasets (small m) never satisfy the strict amax criterion.
    for _ in range(max_passes - 1):
        beta_prev = beta_gls
        psi_diag_prev = Psi_lap.diagonal(dim1=-2, dim2=-1)

        Psi_inv = _ridgeInv(Psi_lap, psi_0_floor)
        beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls, blup_kh_var = _pqlPass(
            beta_gls, Psi_inv, *pass_args, bg_init=blups
        )
        beta_gls, Psi_lap = _sanitize(beta_gls, Psi_lap)
        if uncorr is not None:
            Psi_lap = torch.where(
                uncorr[:, None, None], torch.diag_embed(Psi_lap.diagonal(dim1=-2, dim2=-1)), Psi_lap
            )

        d_beta = (beta_gls - beta_prev).abs().amax(dim=-1)                               # (B,)
        d_psi = (Psi_lap.diagonal(dim1=-2, dim2=-1) - psi_diag_prev).abs().amax(dim=-1)  # (B,)
        if torch.quantile(d_beta, 0.95) < 1e-3 and torch.quantile(d_psi, 0.95) < 1e-3:
            break

    # ------------------------------------------------------------------
    # Pack outputs
    # ------------------------------------------------------------------
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    Psi_lap = Psi_lap.nan_to_num(nan=0.0, posinf=0.0)
    if likelihood_family == 1:
        corr_alpha = G / (G + _BERNOULLI_CORR_SHRINKAGE_C)
        Psi_lap = _shrinkOffDiagonal(Psi_lap, corr_alpha)
    elif likelihood_family == 2:
        corr_alpha = G / (G + _POISSON_CORR_SHRINKAGE_C)
        Psi_lap = _shrinkOffDiagonal(Psi_lap, corr_alpha)
    Psi_pql = Psi_pql.nan_to_num(nan=0.0, posinf=0.0)
    mean_Hg_inv = mean_Hg_inv.nan_to_num(nan=0.0, posinf=0.0)
    sigma_rfx_est = Psi_lap.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    # Per-group posterior variance: diagonal of K_g^{-1} = (ZWZ + Ψ̂_Lap^{-1})^{-1}.
    # Uses the GLS-consistent Ψ̂_Lap precision (not the ridge-inflated Newton Ψ⁻¹),
    # mirroring the Normal path's σ²_ε · W_g.  Cap at 25 (std ≤ 5).
    blup_var = Kg_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)  # (B, m, q)
    blup_var = (blup_var * mask_m[:, :, None]).nan_to_num(nan=0.0, posinf=0.0)

    # Inflate blup_var to account for uncertainty in Ψ̂_Lap (same rationale as Normal path).
    # Use G as denominator (no d subtraction) since PQL doesn't have an explicit df formula.
    df_sigma = G.clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma)[:, None, None]
    blup_var = blup_var + blup_kh_var
    if likelihood_family == 1:
        blup_var = blup_var * _BERNOULLI_BLUP_VAR_INFLATION
        if d > 8:
            blup_var = blup_var + _BERNOULLI_HIGH_D_BLUP_VAR_FLOOR * mask_m[:, :, None]

    # Per-group mean working residual (after removing fixed effects)
    resid_g = (resid_gls.sum(dim=2) / ns.clamp(min=1.0) * mask_m).unsqueeze(-1)  # (B, m, 1)
    resid_g = resid_g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    beta_wg_out = beta_0.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # pooled IRLS (no rfx)
    bhat_out = bhat_ols.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)   # (B, m, q)

    return {
        'beta_est': beta_gls,  # (B, d)
        'beta_wg': beta_wg_out,  # (B, d)  pooled IRLS (no-rfx analog of beta_wg)
        'sigma_rfx_est': sigma_rfx_est,  # (B, q)
        'blup_est': blups,  # (B, m, q)
        'blup_var': blup_var,  # (B, m, q)
        'bhat': bhat_out,  # (B, m, q)
        'resid_g': resid_g,  # (B, m, 1)
        'phi_pearson': phi_pearson,  # (B,)
        'psi_0': psi_0,  # (B,)
        'Psi_pql': Psi_pql,  # (B, q, q)
        'Psi_lap': Psi_lap,  # (B, q, q)
        'mean_Hg_inv': mean_Hg_inv,  # (B, q, q)
    }


def lmmBernoulli(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    n_newton: int = 3,
    uncorr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM for Bernoulli/logit outcomes."""
    return _lmmGlmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=1,
        n_newton=n_newton,
        uncorr=uncorr,
    )


def lmmPoisson(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    n_newton: int = 3,
    uncorr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM for Poisson/log outcomes."""
    return _lmmGlmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=2,
        n_newton=n_newton,
        uncorr=uncorr,
    )
