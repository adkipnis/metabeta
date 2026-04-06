"""LMM/GLMM variance-component estimators: lmmNormal (Gaussian),
lmmBernoulli (logit), lmmPoisson (log). All return the same dict keys.
"""

import torch
import torch.nn.functional as F

from metabeta.utils.glm import _adaptiveRidge, irlsBernoulliCompacted, irlsPoissonCompacted


def _adaptiveRidgeBm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-adaptive ridge for (B, m, q, q) block matrices."""
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)  # (B, m)
    q = A.shape[-1]
    return eps * scale[..., None, None] * torch.eye(q, device=A.device, dtype=A.dtype)


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


# ---------------------------------------------------------------------------
# Private normal-LMM implementations
# ---------------------------------------------------------------------------


def _lmmNormalCompacted(
    Xm: torch.Tensor,       # (B, m, n, d)
    ym: torch.Tensor,       # (B, m, n)
    mask_n: torch.Tensor,   # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,   # (B, m)     1 for active groups
    ns: torch.Tensor,       # (B, m)     group sizes (float, ≥ 1 for active)
    n_total: torch.Tensor,  # (B,)       total active observations
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the random-intercept linear mixed model (q=1).

    Returns a dict with keys: beta_est, sigma_eps_est, sigma_rfx_est, blup_est, Psi.
    """
    B, m, n, d = Xm.shape
    ns_f = ns.clamp(min=1.0)                          # (B, m)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,) active groups
    N = n_total.float()                               # (B,)

    # ------------------------------------------------------------------
    # Stage 1: within-group (WG) OLS for σ_ε
    # ------------------------------------------------------------------
    y_mean = ym.sum(dim=2) / ns_f                                    # (B, m)
    X_mean = Xm.sum(dim=2) / ns_f.unsqueeze(-1)                      # (B, m, d)

    y_tilde = (ym - y_mean.unsqueeze(2)) * mask_n                    # (B, m, n)
    X_tilde = (Xm - X_mean.unsqueeze(2)) * mask_n.unsqueeze(3)       # (B, m, n, d)

    XtX_w = torch.einsum('bmnd,bmnk->bdk', X_tilde, X_tilde)         # (B, d, d)
    Xty_w = torch.einsum('bmnd,bmn->bd', X_tilde, y_tilde)           # (B, d)
    beta_wg = torch.linalg.solve(XtX_w + _adaptiveRidge(XtX_w), Xty_w)

    yhat_w = torch.einsum('bmnd,bd->bmn', X_tilde, beta_wg)
    ss_w = ((y_tilde - yhat_w) ** 2).sum(dim=(1, 2))                 # (B,)
    df_w = (N - G - d + 1).clamp(min=1.0)
    sigma_eps_sq = ss_w / df_w                                        # (B,)

    # ------------------------------------------------------------------
    # Stage 2: between-group MoM (Henderson Method 3) for σ_b
    # ------------------------------------------------------------------
    wt = ns_f * mask_m                                                # (B, m)
    XWX = torch.einsum('bmd,bm,bmk->bdk', X_mean, wt, X_mean)        # (B, d, d)
    XWy = torch.einsum('bmd,bm,bm->bd', X_mean, wt, y_mean * mask_m)
    beta_bg = torch.linalg.solve(XWX + _adaptiveRidge(XWX), XWy)

    resid_bg = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta_bg)) * mask_m
    ss_bg = (wt * resid_bg.square()).sum(dim=1)                       # (B,)
    ms_bg = ss_bg / (G - d).clamp(min=1.0)                           # (B,)

    # Cochran n_0: effective group size for unbalanced designs
    ns_sq = (ns_f.square() * mask_m).sum(dim=1)
    n_0 = ((N - ns_sq / N.clamp(min=1)) / (G - 1).clamp(min=1.0)).clamp(min=1.0)

    sigma_rfx_sq = ((ms_bg - sigma_eps_sq) / n_0).clamp(min=0.0)     # (B,)

    # ------------------------------------------------------------------
    # Stage 3: GLS β̂ via Woodbury
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
    beta_gls = torch.linalg.solve(A_gls + _adaptiveRidge(A_gls), b_gls)

    beta_gls = beta_gls.nan_to_num(nan=0.0)
    sigma_eps = sigma_eps_sq.sqrt().unsqueeze(-1).nan_to_num(nan=1.0, posinf=1.0)  # (B, 1)
    sigma_rfx = sigma_rfx_sq.sqrt().unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)  # (B, 1)

    # BLUPs for q=1: λ_g · (ȳ_g − X̄_g β̂)
    sigma_rfx_sq_val = sigma_rfx.squeeze(-1).square()              # (B,)
    sigma_eps_sq_val = sigma_eps.squeeze(-1).square()              # (B,)
    ns_f2 = ns.clamp(min=1.0)
    lambda_g2 = (ns_f2 * sigma_rfx_sq_val[:, None]) / (
        sigma_eps_sq_val[:, None] + ns_f2 * sigma_rfx_sq_val[:, None]
    )
    y_mean2 = ym.sum(dim=2) / ns_f2
    X_mean2 = Xm.sum(dim=2) / ns_f2.unsqueeze(-1)
    r_g = (y_mean2 - torch.einsum('bmd,bd->bm', X_mean2, beta_gls)) * mask_m
    blups = (lambda_g2 * r_g).unsqueeze(-1).nan_to_num(nan=0.0)    # (B, m, 1)

    Psi = sigma_rfx.square().unsqueeze(-1)                          # (B, 1, 1)

    return {
        'beta_est': beta_gls,           # (B, d)
        'sigma_eps_est': sigma_eps,     # (B, 1)
        'sigma_rfx_est': sigma_rfx,     # (B, 1)
        'blup_est': blups,              # (B, m, 1)
        'Psi': Psi,                     # (B, 1, 1)
    }


def _lmmNormalFull(
    Xm: torch.Tensor,       # (B, m, n, d)
    ym: torch.Tensor,       # (B, m, n)
    Zm: torch.Tensor,       # (B, m, n, q)
    mask_n: torch.Tensor,   # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,   # (B, m)     1 for active groups
    ns: torch.Tensor,       # (B, m)     group sizes (float, ≥ 1 for active)
    n_total: torch.Tensor,  # (B,)       total active observations
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the LME y_g = X_g β + Z_g b_g + ε_g, b_g ~ N(0, Ψ).

    Non-iterative: within-Z projection for σ̂_ε, MINQUE-weighted MoM for Ψ̂,
    then Woodbury GLS for β̂.  Handles arbitrary q (including q=1).

    Returns a dict with keys: beta_est, sigma_eps_est, sigma_rfx_est, blup_est, Psi.
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
    # ------------------------------------------------------------------
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)                # (B, m, q, q)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)  # (B, m, q, q)
    ZtZ_inv = torch.linalg.solve(
        ZtZ_safe + _adaptiveRidgeBm(ZtZ_safe), eye_q_bm
    )                                                              # (B, m, q, q)

    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)                  # (B, m, q)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)                # (B, m, q, d)

    Zhat_y = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Zty)         # (B, m, q)
    Zhat_X = torch.einsum('bmqr,bmrd->bmqd', ZtZ_inv, ZtX)       # (B, m, q, d)
    My = (ym - torch.einsum('bmnq,bmq->bmn', Zm, Zhat_y)) * mask_n           # (B, m, n)
    MX = (Xm - torch.einsum('bmnq,bmqd->bmnd', Zm, Zhat_X)) * mask_n[..., None]  # (B, m, n, d)

    MXtMX = torch.einsum('bmnd,bmnk->bdk', MX, MX)               # (B, d, d)
    MXtMy = torch.einsum('bmnd,bmn->bd', MX, My)                  # (B, d)
    beta_wg = torch.linalg.solve(MXtMX + _adaptiveRidge(MXtMX), MXtMy)  # (B, d)

    resid_M = My - torch.einsum('bmnd,bd->bmn', MX, beta_wg)     # (B, m, n)
    ss_w = resid_M.square().sum(dim=(1, 2))                       # (B,)
    df_w = (N - G * q - (d - q)).clamp(min=1.0)                   # (B,)
    sigma_eps_sq = (ss_w / df_w).clamp(min=0.0)                   # (B,)

    # ------------------------------------------------------------------
    # Stage 2: group-level OLS b̂_g and MoM for Ψ
    # ------------------------------------------------------------------
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)                 # (B, d, d)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)                   # (B, d)
    beta_ols = torch.linalg.solve(XtX + _adaptiveRidge(XtX), Xty)  # (B, d)
    resid_full = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_ols)) * mask_n  # (B, m, n)
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid_full)          # (B, m, q)
    bhat = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Ztr)           # (B, m, q)  b̂_g

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat, bhat)        # (B, m, q, q)
    ZtZ_bhat = torch.einsum('bmqp,bmpk->bmqk', ZtZ_safe, bhat_outer)  # (B, m, q, q)

    mask4 = mask_m[:, :, None, None]                              # (B, m, 1, 1)
    sum_ZtZ = (ZtZ_safe * mask4).sum(dim=1)                       # (B, q, q)
    sum_ZtZ_bhat = (ZtZ_bhat * mask4).sum(dim=1)                  # (B, q, q)

    rhs_mom = sum_ZtZ_bhat - sigma_eps_sq[:, None, None] * G[:, None, None] * eye_q
    Psi_raw = torch.linalg.solve(sum_ZtZ + _adaptiveRidge(sum_ZtZ), rhs_mom)  # (B, q, q)

    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    vals, vecs = torch.linalg.eigh(Psi_raw)                      # (B, q), (B, q, q)
    vals = vals.clamp(min=0.0)
    Psi = vecs @ torch.diag_embed(vals) @ vecs.mT                 # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 3: GLS β̂ via Woodbury
    # ------------------------------------------------------------------
    se2 = sigma_eps_sq.clamp(min=1e-12)

    tol_eig = vals.amax(dim=-1, keepdim=True).clamp(min=1e-8) * 1e-6
    inv_vals = torch.where(vals > tol_eig, 1.0 / vals.clamp(min=1e-30), torch.zeros_like(vals))
    Psi_inv = vecs @ torch.diag_embed(inv_vals) @ vecs.mT        # (B, q, q)

    inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe  # (B, m, q, q)
    W_g = torch.linalg.solve(inner + _adaptiveRidgeBm(inner), eye_q_bm)  # (B, m, q, q)
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
    beta_gls = torch.linalg.solve(A_gls + _adaptiveRidge(A_gls), b_gls)  # (B, d)

    beta_gls = beta_gls.nan_to_num(nan=0.0)
    sigma_eps_1d = sigma_eps_sq.sqrt().nan_to_num(nan=1.0, posinf=1.0)  # (B,)
    Psi = Psi.nan_to_num(nan=0.0, posinf=0.0)

    # BLUPs
    resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)           # (B, m, q)
    blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)              # (B, m, q)
    blups = blups.nan_to_num(nan=0.0)

    sigma_rfx = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()  # (B, q)

    return {
        'beta_est': beta_gls,                       # (B, d)
        'sigma_eps_est': sigma_eps_1d.unsqueeze(-1), # (B, 1)
        'sigma_rfx_est': sigma_rfx,                 # (B, q)
        'blup_est': blups,                          # (B, m, q)
        'Psi': Psi,                                 # (B, q, q)
    }


# ---------------------------------------------------------------------------
# Public normal-LMM dispatcher
# ---------------------------------------------------------------------------


def lmmNormal(
    Xm: torch.Tensor,       # (B, m, n, d)
    ym: torch.Tensor,       # (B, m, n)
    Zm: torch.Tensor,       # (B, m, n, q)
    mask_n: torch.Tensor,   # (B, m, n)
    mask_m: torch.Tensor,   # (B, m)
    ns: torch.Tensor,       # (B, m)
    n_total: torch.Tensor,  # (B,)
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the Gaussian LMM. Routes to q=1 compacted or full."""
    if Zm.shape[-1] == 1:
        return _lmmNormalCompacted(Xm, ym, mask_n, mask_m, ns, n_total)
    return _lmmNormalFull(Xm, ym, Zm, mask_n, mask_m, ns, n_total)


# ---------------------------------------------------------------------------
# Private GLMM implementation
# ---------------------------------------------------------------------------


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

    ZWZ_safe = torch.where(active[:, :, None, None], ZWZ, eye_q)
    ZWZ_inv = torch.linalg.solve(ZWZ_safe + _adaptiveRidgeBm(ZWZ_safe), eye_q_bm)  # (B, m, q, q)

    ZtYmMu = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_0) * mask_n)  # (B, m, q)
    bhat_ols = torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZtYmMu) * mask_m[:, :, None]

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat_ols, bhat_ols)       # (B, m, q, q)
    mean_bhat_outer = (bhat_outer * mask4).sum(dim=1) / G[:, None, None]
    mean_ZWZ_inv = (ZWZ_inv * mask4).sum(dim=1) / G[:, None, None]
    Psi_pql = _psdProject(mean_bhat_outer - mean_ZWZ_inv)                 # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 2: damped Newton → Laplace mode b̂_g, Ψ̂_Lap, β̂_GLS, BLUPs
    # ------------------------------------------------------------------
    Psi_inv = _pseudoInverse(Psi_pql)                                     # (B, q, q)

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
        delta = torch.linalg.solve(Hg + _adaptiveRidgeBm(Hg), grad_g)  # (B, m, q)

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

    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv = torch.linalg.solve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4
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
    Kg_inv = torch.linalg.solve(Kg + _adaptiveRidgeBm(Kg), eye_q_bm) * mask4

    ZWX_f = XWZ_f.mT                                                       # (B, m, q, d)
    A_g = XWX_f - torch.einsum(                                            # Schur complement
        'bmdq,bmqk->bmdk', XWZ_f, torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    )
    rhs_g = XWy_f - torch.einsum('bmdq,bmq->bmd', XWZ_f,
                                  torch.einsum('bmqr,bmr->bmq', Kg_inv, ZWy_f))

    sum_A = (A_g * mask4).sum(dim=1)
    beta_gls = torch.linalg.solve(
        sum_A + _adaptiveRidge(sum_A),
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


# ---------------------------------------------------------------------------
# Public GLMM dispatchers
# ---------------------------------------------------------------------------


def lmmBernoulli(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    n_newton: int = 3,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM for Bernoulli/logit outcomes."""
    return _lmmGlmm(Xm, ym, Zm, mask_n, mask_m, ns, n_total, likelihood_family=1, n_newton=n_newton)


def lmmPoisson(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    n_newton: int = 3,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM for Poisson/log outcomes."""
    return _lmmGlmm(Xm, ym, Zm, mask_n, mask_m, ns, n_total, likelihood_family=2, n_newton=n_newton)


def glmm(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    likelihood_family: int = 0,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Dispatch to lmmNormal / lmmBernoulli / lmmPoisson by likelihood_family."""
    if likelihood_family == 0:
        return lmmNormal(Xm, ym, Zm, mask_n, mask_m, ns, n_total)
    elif likelihood_family == 1:
        return lmmBernoulli(Xm, ym, Zm, mask_n, mask_m, ns, n_total, **kwargs)
    elif likelihood_family == 2:
        return lmmPoisson(Xm, ym, Zm, mask_n, mask_m, ns, n_total, **kwargs)
    raise ValueError(f'unsupported likelihood_family={likelihood_family}')
