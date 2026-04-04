"""Closed-form LME variance component estimators for linear mixed models.

Two variants, both non-iterative (O(N p²) cost):

glsNormalCompacted  — random intercept (q=1) only; fast scalar arithmetic.

glsNormalFull  — arbitrary q; uses within-Z projection for σ̂_ε and a
    MINQUE-weighted MoM estimator for the full Ψ (q×q), then Woodbury GLS.

Both share the three-stage structure:
  Stage 1  Within-group projection → unbiased σ̂_ε
  Stage 2  Group-level b̂_g outer products → Ψ̂ via Henderson MoM
  Stage 3  Woodbury GLS → β̂_GLS using estimated variance components
"""

import torch

from metabeta.utils.least_squares import _adaptive_ridge


def _adaptive_ridge_bm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-adaptive ridge for (B, m, q, q) block matrices."""
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)  # (B, m)
    q = A.shape[-1]
    return eps * scale[..., None, None] * torch.eye(q, device=A.device, dtype=A.dtype)


def glsNormalCompacted(
    Xm: torch.Tensor,       # (B, m, n, d)
    ym: torch.Tensor,       # (B, m, n)
    mask_n: torch.Tensor,   # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,   # (B, m)     1 for active groups
    ns: torch.Tensor,       # (B, m)     group sizes (float, ≥ 1 for active)
    n_total: torch.Tensor,  # (B,)       total active observations
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form GLS for the random-intercept linear mixed model.

    Returns
    -------
    beta      (B, d)  GLS fixed-effect estimates
    sigma_eps (B, 1)  residual SD  (within-group OLS, unbiased)
    sigma_rfx (B, 1)  random-effects SD  (Henderson MoM, truncated at 0)
    """
    B, m, n, d = Xm.shape
    ns_f = ns.clamp(min=1.0)                          # (B, m)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,) active groups
    N = n_total.float()                               # (B,)

    # ------------------------------------------------------------------
    # Stage 1: within-group (WG) OLS for σ_ε
    #
    # Demean per group to project out random intercepts, then regress.
    # df = N - G - d  (G groups absorbed, d slopes estimated).
    # ------------------------------------------------------------------
    y_mean = ym.sum(dim=2) / ns_f                                    # (B, m)
    X_mean = Xm.sum(dim=2) / ns_f.unsqueeze(-1)                      # (B, m, d)

    y_tilde = (ym - y_mean.unsqueeze(2)) * mask_n                    # (B, m, n)
    X_tilde = (Xm - X_mean.unsqueeze(2)) * mask_n.unsqueeze(3)       # (B, m, n, d)

    XtX_w = torch.einsum('bmnd,bmnk->bdk', X_tilde, X_tilde)         # (B, d, d)
    Xty_w = torch.einsum('bmnd,bmn->bd', X_tilde, y_tilde)           # (B, d)
    beta_wg = torch.linalg.solve(XtX_w + _adaptive_ridge(XtX_w), Xty_w)

    yhat_w = torch.einsum('bmnd,bd->bmn', X_tilde, beta_wg)
    ss_w = ((y_tilde - yhat_w) ** 2).sum(dim=(1, 2))                 # (B,)
    df_w = (N - G - d + 1).clamp(min=1.0)
    sigma_eps_sq = ss_w / df_w                                        # (B,)

    # ------------------------------------------------------------------
    # Stage 2: between-group MoM (Henderson Method 3) for σ_b
    #
    # Regress group means with weights n_g, compute MS_BG, then subtract
    # the σ_ε contribution scaled by Cochran's effective group size n_0.
    # ------------------------------------------------------------------
    wt = ns_f * mask_m                                                # (B, m)
    XWX = torch.einsum('bmd,bm,bmk->bdk', X_mean, wt, X_mean)        # (B, d, d)
    XWy = torch.einsum('bmd,bm,bm->bd', X_mean, wt, y_mean * mask_m)
    beta_bg = torch.linalg.solve(XWX + _adaptive_ridge(XWX), XWy)

    resid_bg = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta_bg)) * mask_m
    ss_bg = (wt * resid_bg.square()).sum(dim=1)                       # (B,)
    ms_bg = ss_bg / (G - d).clamp(min=1.0)                           # (B,)

    # Cochran n_0: effective group size for unbalanced designs
    ns_sq = (ns_f.square() * mask_m).sum(dim=1)
    n_0 = ((N - ns_sq / N.clamp(min=1)) / (G - 1).clamp(min=1.0)).clamp(min=1.0)

    sigma_rfx_sq = ((ms_bg - sigma_eps_sq) / n_0).clamp(min=0.0)     # (B,)

    # ------------------------------------------------------------------
    # Stage 3: GLS β̂ via Woodbury
    #
    # V_g^{-1} = (1/σ_ε²)[I - (λ_g/n_g) 1 1ᵀ]
    # where λ_g = n_g σ_b² / (σ_ε² + n_g σ_b²)  (per-group shrinkage).
    #
    # X'V^{-1}X = (1/σ_ε²)[XᵀX − Σ_g λ_g n_g X̄_g X̄_gᵀ]
    # X'V^{-1}y = (1/σ_ε²)[Xᵀy − Σ_g λ_g n_g X̄_g ȳ_g]
    # ------------------------------------------------------------------
    se2 = sigma_eps_sq.clamp(min=1e-12)
    lambda_g = (ns_f * sigma_rfx_sq[:, None]) / (se2[:, None] + ns_f * sigma_rfx_sq[:, None])
    lambda_g = lambda_g * mask_m                                      # (B, m)

    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)                     # (B, d, d)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)                        # (B, d)
    w_bg = lambda_g * ns_f                                            # (B, m)
    XbarXbar = torch.einsum('bmd,bm,bmk->bdk', X_mean, w_bg, X_mean)  # (B, d, d)
    Xbary = torch.einsum('bmd,bm,bm->bd', X_mean, w_bg, y_mean)

    inv_se2 = 1.0 / se2
    A_gls = inv_se2[:, None, None] * (XtX - XbarXbar)
    b_gls = inv_se2[:, None] * (Xty - Xbary)
    beta_gls = torch.linalg.solve(A_gls + _adaptive_ridge(A_gls), b_gls)

    beta_gls = beta_gls.nan_to_num(nan=0.0)
    sigma_eps = sigma_eps_sq.sqrt().unsqueeze(-1).nan_to_num(nan=1.0, posinf=1.0)
    sigma_rfx = sigma_rfx_sq.sqrt().unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)
    return beta_gls, sigma_eps, sigma_rfx


def glsNormalFull(
    Xm: torch.Tensor,       # (B, m, n, d)
    ym: torch.Tensor,       # (B, m, n)
    Zm: torch.Tensor,       # (B, m, n, q)
    mask_n: torch.Tensor,   # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,   # (B, m)     1 for active groups
    ns: torch.Tensor,       # (B, m)     group sizes (float, ≥ 1 for active)
    n_total: torch.Tensor,  # (B,)       total active observations
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form GLS for the LME y_g = X_g β + Z_g b_g + ε_g, b_g ~ N(0, Ψ).

    Non-iterative: within-Z projection for σ̂_ε, MINQUE-weighted MoM for Ψ̂,
    then Woodbury GLS for β̂.  Handles arbitrary q (including q=1).

    Returns
    -------
    beta      (B, d)    GLS fixed-effect estimates
    sigma_eps (B,)      residual SD  (within-Z OLS, unbiased)
    Psi       (B, q, q) random-effects covariance matrix  (PSD)
    """
    B, m, n, d = Xm.shape
    q = Zm.shape[-1]
    N = n_total.float()                               # (B,)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,)
    active = mask_m.bool()                            # (B, m)

    eye_q = torch.eye(q, device=Xm.device, dtype=Xm.dtype)       # (q, q)
    eye_q_bm = eye_q.expand(B, m, q, q)                          # (B, m, q, q)

    # ------------------------------------------------------------------
    # Stage 1: within-Z projection → σ̂_ε
    #
    # M_g = I - Z_g (Z_g^T Z_g)^{-1} Z_g^T  (within-Z annihilator per group)
    # Project both y and X through M, then run OLS on projected data.
    # df = N - G*q - (d - q)  (G*q random-effect dof + d-q fixed-effect slopes)
    # ------------------------------------------------------------------
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)                # (B, m, q, q)
    # Replace padded groups' ZtZ with I so the solve stays non-singular.
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)  # (B, m, q, q)
    ZtZ_inv = torch.linalg.solve(
        ZtZ_safe + _adaptive_ridge_bm(ZtZ_safe), eye_q_bm
    )                                                              # (B, m, q, q)

    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)                  # (B, m, q)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)                # (B, m, q, d)

    # Projected y and X: remove the Z-subspace per group
    Zhat_y = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Zty)         # (B, m, q)
    Zhat_X = torch.einsum('bmqr,bmrd->bmqd', ZtZ_inv, ZtX)       # (B, m, q, d)
    My = (ym - torch.einsum('bmnq,bmq->bmn', Zm, Zhat_y)) * mask_n           # (B, m, n)
    MX = (Xm - torch.einsum('bmnq,bmqd->bmnd', Zm, Zhat_X)) * mask_n[..., None]  # (B, m, n, d)

    MXtMX = torch.einsum('bmnd,bmnk->bdk', MX, MX)               # (B, d, d)
    MXtMy = torch.einsum('bmnd,bmn->bd', MX, My)                  # (B, d)
    beta_wg = torch.linalg.solve(MXtMX + _adaptive_ridge(MXtMX), MXtMy)  # (B, d)

    resid_M = My - torch.einsum('bmnd,bd->bmn', MX, beta_wg)     # (B, m, n)
    ss_w = resid_M.square().sum(dim=(1, 2))                       # (B,)
    df_w = (N - G * q - (d - q)).clamp(min=1.0)                   # (B,)
    sigma_eps_sq = (ss_w / df_w).clamp(min=0.0)                   # (B,)

    # ------------------------------------------------------------------
    # Stage 2: group-level OLS b̂_g and MoM for Ψ
    #
    # b̂_g = (Z_g^T Z_g)^{-1} Z_g^T (y_g - X_g β_OLS)
    # E[b̂_g b̂_g^T] ≈ Ψ + σ_ε² (Z_g^T Z_g)^{-1}  (up to 1/N corrections)
    #
    # Key: use pooled OLS β̂ to form residuals (not β_WG) because β_WG does
    # not estimate the components of β that lie in the Z column space — those
    # are projected out.  Pooled OLS consistently estimates all fixed effects,
    # so E[b̂_g] ≈ 0 and the outer products are unbiased for Ψ.
    #
    # MINQUE-weighted estimator  (A_g = Z_g^T Z_g):
    #   Ψ̂ = (Σ A_g)^{-1} [Σ ZtZ b̂b̂^T - G σ_ε² I]
    # followed by PSD projection (clip negative eigenvalues).
    # ------------------------------------------------------------------
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)                 # (B, d, d)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)                   # (B, d)
    beta_ols = torch.linalg.solve(XtX + _adaptive_ridge(XtX), Xty)  # (B, d)
    resid_full = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_ols)) * mask_n  # (B, m, n)
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid_full)          # (B, m, q)
    bhat = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Ztr)           # (B, m, q)  b̂_g

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat, bhat)        # (B, m, q, q)
    # ZtZ_safe[inactive] = I, bhat[inactive] = 0, so their contribution is 0.
    ZtZ_bhat = torch.einsum('bmqp,bmpk->bmqk', ZtZ_safe, bhat_outer)  # (B, m, q, q)

    mask4 = mask_m[:, :, None, None]                              # (B, m, 1, 1)
    sum_ZtZ = (ZtZ_safe * mask4).sum(dim=1)                       # (B, q, q)
    sum_ZtZ_bhat = (ZtZ_bhat * mask4).sum(dim=1)                  # (B, q, q)

    rhs_mom = sum_ZtZ_bhat - sigma_eps_sq[:, None, None] * G[:, None, None] * eye_q
    Psi_raw = torch.linalg.solve(sum_ZtZ + _adaptive_ridge(sum_ZtZ), rhs_mom)  # (B, q, q)

    # Symmetrize and project onto PSD cone
    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    vals, vecs = torch.linalg.eigh(Psi_raw)                      # (B, q), (B, q, q)
    vals = vals.clamp(min=0.0)
    Psi = vecs @ torch.diag_embed(vals) @ vecs.mT                 # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 3: GLS β̂ via Woodbury
    #
    # V_g^{-1} = (1/σ_ε²) [I - Z_g (Ψ^{-1} σ_ε² + Z_g^T Z_g)^{-1} Z_g^T]
    # X^T V^{-1} X = (1/σ_ε²) [XtX - Σ_g XtZ_g W_g ZtX_g]
    # X^T V^{-1} y = (1/σ_ε²) [Xty - Σ_g XtZ_g W_g Zty_g]
    # where W_g = (Ψ^{-1} σ_ε² + Z_g^T Z_g)^{-1}  (q×q per group)
    # ------------------------------------------------------------------
    se2 = sigma_eps_sq.clamp(min=1e-12)

    # Safe Ψ^{-1} via eigendecomposition (pseudo-inverse for zero eigenvalues)
    tol_eig = vals.amax(dim=-1, keepdim=True).clamp(min=1e-8) * 1e-6
    inv_vals = torch.where(vals > tol_eig, 1.0 / vals.clamp(min=1e-30), torch.zeros_like(vals))
    Psi_inv = vecs @ torch.diag_embed(inv_vals) @ vecs.mT        # (B, q, q)

    # W_g = (Ψ^{-1} σ_ε² + ZtZ_g)^{-1}  — inactive groups: ZtZ_safe = I, Ψ^{-1}≥0
    inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe  # (B, m, q, q)
    W_g = torch.linalg.solve(inner + _adaptive_ridge_bm(inner), eye_q_bm)  # (B, m, q, q)

    # Zero out contributions from inactive groups
    W_g = W_g * mask4                                             # (B, m, q, q)

    XtX_pool = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)            # (B, d, d)
    Xty_pool = torch.einsum('bmnd,bmn->bd', Xm, ym)              # (B, d)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)               # (B, m, d, q)

    W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)            # (B, m, q, d)
    correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)   # (B, d, d)
    W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)              # (B, m, q)
    correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)     # (B, d)

    inv_se2 = 1.0 / se2
    A_gls = inv_se2[:, None, None] * (XtX_pool - correction_XX)  # (B, d, d)
    b_gls = inv_se2[:, None] * (Xty_pool - correction_Xy)        # (B, d)
    beta_gls = torch.linalg.solve(A_gls + _adaptive_ridge(A_gls), b_gls)  # (B, d)

    beta_gls = beta_gls.nan_to_num(nan=0.0)
    sigma_eps = sigma_eps_sq.sqrt().nan_to_num(nan=1.0, posinf=1.0)
    Psi = Psi.nan_to_num(nan=0.0, posinf=0.0)

    # ------------------------------------------------------------------
    # BLUPs: b̂_g = K_g^{-1} Z_g^T (y_g − X_g β̂_GLS)
    #
    # W_g = K_g^{-1} is already computed above; BLUPs cost only a Z^T r
    # inner product plus the (B, m, q, q) × (B, m, q) matmul.
    #
    # Posterior covariance: Cov(b_g | y_g) = σ_ε² K_g^{-1} = σ_ε² W_g
    # (derived by substituting the Woodbury V_g^{-1} into the standard
    # conditional Gaussian formula; the Ψ factor cancels exactly).
    # ------------------------------------------------------------------
    resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)           # (B, m, q)
    blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)              # (B, m, q)
    blups = blups.nan_to_num(nan=0.0)

    return beta_gls, sigma_eps, Psi, blups


def glsNormal(
    Xm: torch.Tensor,       # (B, m, n, d)
    ym: torch.Tensor,       # (B, m, n)
    Zm: torch.Tensor,       # (B, m, n, q)
    mask_n: torch.Tensor,   # (B, m, n)
    mask_m: torch.Tensor,   # (B, m)
    ns: torch.Tensor,       # (B, m)
    n_total: torch.Tensor,  # (B,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to q=1 or full GLS depending on the Z dimension.

    Returns
    -------
    beta      (B, d)    GLS fixed-effect estimates
    sigma_eps (B, 1)    residual SD
    sigma_rfx (B, q)    marginal random-effect SDs  (sqrt of diag(Ψ̂))
    blups     (B, m, q) BLUP estimates of group random effects
    """
    q = Zm.shape[-1]
    if q == 1:
        beta, sigma_eps, sigma_rfx = glsNormalCompacted(
            Xm, ym, mask_n, mask_m, ns, n_total
        )
        # BLUPs for q=1: λ_g · (ȳ_g − X̄_g β̂)  (shrinkage of group-mean residual)
        # λ_g = n_g σ_b² / (σ_e² + n_g σ_b²) is the scalar shrinkage factor.
        sigma_rfx_sq = sigma_rfx.squeeze(-1).square()           # (B,)
        sigma_eps_sq = sigma_eps.squeeze(-1).square()           # (B,)
        ns_f = ns.clamp(min=1.0)
        lambda_g = (
            ns_f * sigma_rfx_sq[:, None]
            / (sigma_eps_sq[:, None] + ns_f * sigma_rfx_sq[:, None])
        )                                                        # (B, m)
        y_mean = ym.sum(dim=2) / ns_f                           # (B, m)
        X_mean = Xm.sum(dim=2) / ns_f.unsqueeze(-1)            # (B, m, d)
        r_g = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta)) * mask_m
        blups = (lambda_g * r_g).unsqueeze(-1).nan_to_num(nan=0.0)  # (B, m, 1)
        return beta, sigma_eps, sigma_rfx, blups
    else:
        beta, sigma_eps_1d, Psi, blups = glsNormalFull(
            Xm, ym, Zm, mask_n, mask_m, ns, n_total
        )
        sigma_rfx = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()  # (B, q)
        return beta, sigma_eps_1d.unsqueeze(-1), sigma_rfx, blups
