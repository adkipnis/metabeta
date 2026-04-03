"""Closed-form LME variance component estimator for linear mixed models.

Two-stage OLS + Henderson Method-3 pipeline (no iterative solver):

  Stage 1  Within-group (WG) OLS → unbiased σ̂_ε
  Stage 2  Between-group method of moments → unbiased σ̂_b (Henderson 3)
  Stage 3  Woodbury GLS → β̂_GLS using estimated variance components

Total cost: O(N p²), same order as pooled OLS.
"""

import torch

from metabeta.utils.least_squares import _adaptive_ridge


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
