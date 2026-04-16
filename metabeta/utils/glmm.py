"""LMM/GLMM variance-component estimators: lmmNormal (Gaussian),
lmmBernoulli (logit), lmmPoisson (log). All return the same dict keys.
"""

import torch
import torch.nn.functional as F

from metabeta.utils.glm import _adaptiveRidge, _safeSolve, irlsBernoulliCompacted, irlsPoissonCompacted


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
    n_em: int = 3,
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the random-intercept linear mixed model (q=1).

    Returns a dict with keys: beta_est, sigma_eps_est, sigma_rfx_est, blup_est, blup_var, Psi.
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
    beta_wg = _safeSolve(XtX_w + _adaptiveRidge(XtX_w), Xty_w)

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
    beta_bg = _safeSolve(XWX + _adaptiveRidge(XWX), XWy)

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
    beta_gls = _safeSolve(A_gls + _adaptiveRidge(A_gls), b_gls)

    # clamp before BLUP computation: near-cancellation in A_gls=(XtX−XbarXbar) when λ→1
    # produces finite-but-huge values that nan_to_num cannot catch.
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
    sigma_eps = sigma_eps_sq.sqrt().unsqueeze(-1).nan_to_num(nan=1.0, posinf=1.0)  # (B, 1)
    sigma_rfx = sigma_rfx_sq.sqrt().unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)  # (B, 1)

    # BLUPs for q=1: EM-refined shrinkage.
    sigma_rfx_sq_val = sigma_rfx.squeeze(-1).square()              # (B,)  MoM, may be 0
    sigma_eps_sq_val = sigma_eps.squeeze(-1).square()              # (B,)

    r_g = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta_gls)) * mask_m

    # EM init: floor σ_rfx² with 10% of mean(r_g²) so EM doesn't get stuck when MoM clips
    # to 0. mean_g(r_g²) ≈ σ_eps²/n̄ + σ_rfx² in expectation — always ≥ 0.
    r_g_var = (r_g.square() * mask_m).sum(dim=1) / G              # (B,)
    # Floor σ_rfx² at 50% of r_g_var so EM doesn't get stuck when MoM clips to 0.
    # E[r_g²] ≈ σ_rfx² + σ_ε²/n̄, so 0.5 × r_g_var ≈ σ_rfx²/2 at high SNR — close
    # enough that EM converges in 1 step (λ_g → 1 → BLUPs ≈ r_g → M-step ≈ mean(r_g²)).
    sigma_rfx_sq_val = sigma_rfx_sq_val.clamp(min=r_g_var * 0.5)

    lambda_g2 = (ns_f * sigma_rfx_sq_val[:, None]) / (
        sigma_eps_sq_val[:, None] + ns_f * sigma_rfx_sq_val[:, None]
    )
    blups = (lambda_g2 * r_g).unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # (B, m, 1)

    # Posterior variance: Var(b_g | data) = σ_rfx² · (1 − λ_g)
    blup_var = (sigma_rfx_sq_val[:, None] * (1.0 - lambda_g2)).clamp(min=0.0).unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)  # (B, m, 1)

    # EM iterations: jointly update σ_rfx², σ_ε², and β̂_GLS.
    # M-step: σ_rfx² = mean_g(b̂_g² + σ²_rfx(1−λ_g)), σ_ε² = RSS/(N−d−T) (REML-like).
    # E-step: recompute λ_g, then β̂_GLS, r_g, BLUPs under updated parameters.
    for _ in range(n_em):
        # M-step
        sigma_rfx_sq_val = (
            (blups.squeeze(-1).square() + blup_var.squeeze(-1)) * mask_m
        ).sum(dim=1) / G
        sigma_rfx_sq_val = sigma_rfx_sq_val.clamp(min=0.0)

        fitted = torch.einsum('bmnd,bd->bmn', Xm, beta_gls) + blups.squeeze(-1).unsqueeze(-1)
        ss_em = ((ym - fitted) * mask_n).square().sum(dim=(1, 2))   # (B,)
        T = (lambda_g2 * mask_m).sum(dim=1)                         # (B,)  Σ_g λ_g
        # Cap T so the REML denominator N−d−T stays ≥ 10% of N−d, preventing blow-up
        # when T ≈ N−d (all λ_g → 1, i.e., large rfx relative to noise).
        T_safe = T.clamp(max=0.9 * (N - d).clamp(min=1.0))
        sigma_eps_sq_val = (ss_em / (N - d - T_safe).clamp(min=1.0)).clamp(min=1e-12)

        # E-step
        lambda_g2 = (ns_f * sigma_rfx_sq_val[:, None]) / (
            sigma_eps_sq_val[:, None] + ns_f * sigma_rfx_sq_val[:, None]
        )
        w_bg = lambda_g2 * ns_f
        XbarXbar = torch.einsum('bmd,bm,bmk->bdk', X_mean, w_bg, X_mean)
        Xbary = torch.einsum('bmd,bm,bm->bd', X_mean, w_bg, y_mean)
        inv_se2 = 1.0 / sigma_eps_sq_val
        A_gls = inv_se2[:, None, None] * (XtX - XbarXbar)
        b_gls = inv_se2[:, None] * (Xty - Xbary)
        beta_gls = _safeSolve(A_gls + _adaptiveRidge(A_gls), b_gls)
        beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        r_g = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta_gls)) * mask_m
        blups = (lambda_g2 * r_g).unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        blup_var = (sigma_rfx_sq_val[:, None] * (1.0 - lambda_g2)).clamp(min=0.0).unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)

    sigma_rfx = sigma_rfx_sq_val.clamp(min=0.0).sqrt().unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)
    Psi = sigma_rfx.square().unsqueeze(-1)                          # (B, 1, 1)
    sigma_eps = sigma_eps_sq_val.clamp(min=0.0).sqrt().unsqueeze(-1).nan_to_num(nan=1.0, posinf=1.0)

    return {
        'beta_est': beta_gls,           # (B, d)
        'sigma_eps_est': sigma_eps,     # (B, 1)
        'sigma_rfx_est': sigma_rfx,     # (B, 1)
        'blup_est': blups,              # (B, m, 1)
        'blup_var': blup_var,           # (B, m, 1)
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
    n_em: int = 3,
) -> dict[str, torch.Tensor]:
    """GLS estimator for the LME y_g = X_g β + Z_g b_g + ε_g, b_g ~ N(0, Ψ).

    Three-stage pipeline: (1) within-Z projection for σ̂_ε, (2) MoM for Ψ̂,
    (3) Woodbury GLS for β̂ and BLUPs; followed by EM refinement of Ψ and σ_ε.
    Handles arbitrary q (including q=1).

    Returns a dict with keys: beta_est, sigma_eps_est, sigma_rfx_est, blup_est, blup_var, Psi.
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
    ZtZ_inv = _safeSolve(
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
    beta_wg = _safeSolve(MXtMX + _adaptiveRidge(MXtMX), MXtMy)  # (B, d)

    resid_M = My - torch.einsum('bmnd,bd->bmn', MX, beta_wg)     # (B, m, n)
    ss_w = resid_M.square().sum(dim=(1, 2))                       # (B,)
    df_w = (N - G * q - (d - q)).clamp(min=1.0)                   # (B,)
    sigma_eps_sq = (ss_w / df_w).clamp(min=0.0)                   # (B,)

    # ------------------------------------------------------------------
    # Stage 2: group-level OLS b̂_g and MoM for Ψ
    # ------------------------------------------------------------------
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)                 # (B, d, d)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)                   # (B, d)
    beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)  # (B, d)
    resid_full = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_ols)) * mask_n  # (B, m, n)
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid_full)          # (B, m, q)
    bhat = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Ztr)           # (B, m, q)  b̂_g

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat, bhat)        # (B, m, q, q)
    ZtZ_bhat = torch.einsum('bmqp,bmpk->bmqk', ZtZ_safe, bhat_outer)  # (B, m, q, q)

    mask4 = mask_m[:, :, None, None]                              # (B, m, 1, 1)
    sum_ZtZ = (ZtZ_safe * mask4).sum(dim=1)                       # (B, q, q)
    sum_ZtZ_bhat = (ZtZ_bhat * mask4).sum(dim=1)                  # (B, q, q)

    rhs_mom = sum_ZtZ_bhat - sigma_eps_sq[:, None, None] * G[:, None, None] * eye_q
    Psi_raw = _safeSolve(sum_ZtZ + _adaptiveRidge(sum_ZtZ), rhs_mom)  # (B, q, q)

    # Per-component noise-corrected Psi diagonal floor to prevent EM getting stuck near Psi=0.
    # E[bhat_i²] ≈ Psi_ii + σ_ε² mean_g(ZtZ_inv_ii). Removing the noise term gives an
    # estimate of Psi_ii ≈ σ_rfx_i², which is near 0 when σ_rfx_i is small (so the floor
    # only activates for high-SNR components where MoM tends to under-estimate).
    # Applied per-component so inactive rfx dimensions (second Z column = 0 for q=1 datasets)
    # are not inflated — their signal_var ≈ 0 and the floor stays at 0.
    mean_ZtZ_inv_diag = (
        ZtZ_inv.diagonal(dim1=-2, dim2=-1) * mask_m[:, :, None]
    ).sum(dim=1) / G[:, None]                                           # (B, q)
    bhat_var = (bhat.square() * mask_m[:, :, None]).sum(dim=1) / G[:, None]  # (B, q)
    psi_diag_floor = (bhat_var - sigma_eps_sq[:, None] * mean_ZtZ_inv_diag).clamp(min=0.0) * 0.5
    Psi_raw = Psi_raw + torch.diag_embed(
        (psi_diag_floor - Psi_raw.diagonal(dim1=-2, dim2=-1)).clamp(min=0.0)
    )                                                                   # bump diag to floor

    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    vals, vecs = torch.linalg.eigh(Psi_raw)                      # (B, q), (B, q, q)
    vals = vals.clamp(min=0.0)
    Psi = vecs @ torch.diag_embed(vals) @ vecs.mT                 # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 3: GLS β̂ via Woodbury
    # ------------------------------------------------------------------
    se2 = sigma_eps_sq.clamp(min=1e-12)

    # Ridge-regularized Psi_inv: (Psi + σ_ε²×1e-4×I)^{-1} reusing Stage-2 eigenvectors.
    # Pseudoinverse zeros eigenvalues ≈ 0 → Psi_inv≈0 → inner≈ZtZ → W_g=ZtZ_inv (OLS, no shrinkage).
    # Ridge ensures inv is large for small eigenvalues → strong shrinkage → BLUPs→0 for small Psi.
    # Effect is negligible when Psi eigenvalues >> σ_ε²×1e-4.
    inv_vals = 1.0 / (vals + se2[:, None] * 1e-4).clamp(min=1e-30)
    Psi_inv = vecs @ torch.diag_embed(inv_vals) @ vecs.mT        # (B, q, q)

    inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe  # (B, m, q, q)
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm)  # (B, m, q, q)
    W_g = W_g * mask4                                             # (B, m, q, q)

    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)                  # (B, m, d, q)

    W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)              # (B, m, q, d)
    correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)     # (B, d, d)
    W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)                # (B, m, q)
    correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)       # (B, d)

    inv_se2 = 1.0 / se2
    A_gls = inv_se2[:, None, None] * (XtX - correction_XX)         # (B, d, d)
    b_gls = inv_se2[:, None] * (Xty - correction_Xy)               # (B, d)
    beta_gls = _safeSolve(A_gls + _adaptiveRidge(A_gls), b_gls)  # (B, d)

    # clamp before BLUP computation: near-cancellation in A_gls=(XtX−correction_XX) when
    # Psi_lap≈0 produces finite-but-huge values that nan_to_num cannot catch.
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
    Psi = Psi.nan_to_num(nan=0.0, posinf=0.0)

    # BLUPs
    resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)           # (B, m, q)
    blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)              # (B, m, q)
    # clamp: W_g can be huge when ZtZ is near-singular (small n_g, q=2) even with beta clamped
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    # Posterior variance: Cov(b_g | data) = σ_ε² · W_g  (diagonal = marginal var)
    # cap at 25 (std ≤ 5) — W_g can be large when ZtZ near-singular (small n_g, q=2)
    blup_var = (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var.nan_to_num(nan=0.0, posinf=0.0)              # (B, m, q)

    # EM iterations: jointly update Ψ, σ_ε², and β̂_GLS.
    # M-step: Ψ = mean_g(b̂_g b̂_g' + σ²_ε W_g) — exact E[b_g b_g'|data] for Gaussian b_g,
    #         σ_ε² = RSS/(N−d−T) (REML-like, T = Σ_g tr(ZtZ_g W_g) effective df).
    # E-step: W_g via ridge-regularized Ψ⁻¹, then β̂_GLS and BLUPs under updated parameters.
    for _ in range(n_em):
        # M-step: Ψ using full posterior covariance (exact for Gaussian b_g)
        blup_outer = torch.einsum('bmq,bmr->bmqr', blups, blups)     # (B, m, q, q)
        post_cov = se2[:, None, None, None] * W_g                    # (B, m, q, q)  σ²_ε W_g
        Psi = _psdProject(
            ((blup_outer + post_cov) * mask4).sum(dim=1) / G[:, None, None]
        )  # (B, q, q)

        # M-step: σ_ε² (REML-like df correction using current blups and beta_gls)
        resid_em = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)
                    - torch.einsum('bmnq,bmq->bmn', Zm, blups)) * mask_n
        ss_em = resid_em.square().sum(dim=(1, 2))                    # (B,)
        ZtZ_W = torch.einsum('bmqr,bmrs->bmqs', ZtZ_safe, W_g)      # (B, m, q, q)
        T = (ZtZ_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1) * mask_m).sum(dim=1)  # (B,)
        # Cap T so the REML denominator N−d−T stays ≥ 10% of N−d, preventing blow-up
        # when T ≈ N−d (λ_g → 1 for all groups at high SNR or near-singular ZtZ_g).
        T_safe = T.clamp(max=0.9 * (N - d).clamp(min=1.0))
        se2 = (ss_em / (N - d - T_safe).clamp(min=1.0)).clamp(min=1e-12)

        # E-step: W_g via ridge-regularized Ψ⁻¹ using updated Ψ and se2
        psi_ridge = se2[:, None, None] * 1e-4 * eye_q
        psi_reg = Psi + psi_ridge
        vals_r, vecs_r = torch.linalg.eigh(psi_reg)
        Psi_inv = vecs_r @ torch.diag_embed(1.0 / vals_r.clamp(min=1e-30)) @ vecs_r.mT
        inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe  # (B, m, q, q)
        W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm) * mask4

        # E-step: β̂_GLS and Ztr_gls under updated W_g and se2
        inv_se2 = 1.0 / se2
        W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)
        correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)
        W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)
        correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)
        A_gls = inv_se2[:, None, None] * (XtX - correction_XX)
        b_gls = inv_se2[:, None] * (Xty - correction_Xy)
        beta_gls = _safeSolve(A_gls + _adaptiveRidge(A_gls), b_gls)
        beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
        Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)

        # E-step: BLUPs and posterior variance
        blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        blup_var = (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
        blup_var = blup_var.nan_to_num(nan=0.0, posinf=0.0)

    sigma_rfx = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()          # (B, q)
    sigma_eps_1d = se2.clamp(min=0.0).sqrt().nan_to_num(nan=1.0, posinf=1.0)  # (B,)

    return {
        'beta_est': beta_gls,                       # (B, d)
        'sigma_eps_est': sigma_eps_1d.unsqueeze(-1), # (B, 1)
        'sigma_rfx_est': sigma_rfx,                 # (B, q)
        'blup_est': blups,                          # (B, m, q)
        'blup_var': blup_var,                       # (B, m, q)
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
    n_em: int = 3,
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the Gaussian LMM. Routes to q=1 compacted or full."""
    if Zm.shape[-1] == 1:
        return _lmmNormalCompacted(Xm, ym, mask_n, mask_m, ns, n_total, n_em=n_em)
    return _lmmNormalFull(Xm, ym, Zm, mask_n, mask_m, ns, n_total, n_em=n_em)


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
    ZWZ_inv = _safeSolve(ZWZ_safe + _adaptiveRidgeBm(ZWZ_safe), eye_q_bm)  # (B, m, q, q)

    ZtYmMu = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_0) * mask_n)  # (B, m, q)
    bhat_ols = torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZtYmMu) * mask_m[:, :, None]

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat_ols, bhat_ols)       # (B, m, q, q)
    mean_bhat_outer = (bhat_outer * mask4).sum(dim=1) / G[:, None, None]
    mean_ZWZ_inv = (ZWZ_inv * mask4).sum(dim=1) / G[:, None, None]
    Psi_pql = _psdProject(mean_bhat_outer - mean_ZWZ_inv)                 # (B, q, q)

    # Floor eigenvalues at psi_0 (overdispersion-based RE scale estimate).
    # Prevents degenerate near-zero Ψ̂_PQL after PSD projection, which would
    # make Ψ̂_PQL⁻¹ → ∞ and kill Newton updates.
    vals_pql, vecs_pql = torch.linalg.eigh(Psi_pql)
    vals_pql = vals_pql.clamp(min=psi_0[:, None])
    Psi_pql = vecs_pql @ torch.diag_embed(vals_pql) @ vecs_pql.mT        # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 2: damped Newton → Laplace mode b̂_g, Ψ̂_Lap, β̂_GLS, BLUPs
    # ------------------------------------------------------------------
    Psi_inv = _pseudoInverse(Psi_pql)                                     # (B, q, q)

    # Cold-start from zero. Warm-starting from bhat_ols is unsafe for Poisson:
    # large bhat_ols values cause exp(η) overflow, making f(bhat_ols) = inf and
    # stalling Armijo (alpha → 0, bg frozen). Cold-start also acts as implicit
    # regularisation: with n_newton=3 the iterate hasn't fully converged under
    # the noisy Psi_pql penalty, providing extra shrinkage that corrects for
    # MoM upward bias after PSD projection. More steps or warm-start worsen BLUPs.
    bg = ym.new_zeros(B, m, q)
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
        delta = _safeSolve(Hg + _adaptiveRidgeBm(Hg), grad_g)  # (B, m, q)

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
    Hg_inv = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4
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
    Kg_inv = _safeSolve(Kg + _adaptiveRidgeBm(Kg), eye_q_bm) * mask4

    ZWX_f = XWZ_f.mT                                                       # (B, m, q, d)
    A_g = XWX_f - torch.einsum(                                            # Schur complement
        'bmdq,bmqk->bmdk', XWZ_f, torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    )
    rhs_g = XWy_f - torch.einsum('bmdq,bmq->bmd', XWZ_f,
                                  torch.einsum('bmqr,bmr->bmq', Kg_inv, ZWy_f))

    sum_A = (A_g * mask4).sum(dim=1)
    beta_gls = _safeSolve(
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
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    Psi_lap = Psi_lap.nan_to_num(nan=0.0, posinf=0.0)
    Psi_pql = Psi_pql.nan_to_num(nan=0.0, posinf=0.0)
    mean_Hg_inv = mean_Hg_inv.nan_to_num(nan=0.0, posinf=0.0)
    sigma_rfx_est = Psi_lap.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    # Per-group posterior variance of each BLUP: diagonal of H_g^{-1}
    blup_var = Hg_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)     # (B, m, q)
    blup_var = (blup_var * mask_m[:, :, None]).nan_to_num(nan=0.0, posinf=0.0)

    return {
        'beta_est': beta_gls,           # (B, d)
        'sigma_rfx_est': sigma_rfx_est, # (B, q)
        'blup_est': blups,              # (B, m, q)
        'blup_var': blup_var,           # (B, m, q)
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
    eta_rfx: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Dispatch to lmmNormal / lmmBernoulli / lmmPoisson by likelihood_family.

    When eta_rfx is provided, datasets with eta_rfx == 0 (uncorrelated rfx) have
    their Psi constrained to diagonal — zeroing noisy off-diagonal MoM estimates.
    """
    if likelihood_family == 0:
        stats = lmmNormal(Xm, ym, Zm, mask_n, mask_m, ns, n_total)
    elif likelihood_family == 1:
        stats = lmmBernoulli(Xm, ym, Zm, mask_n, mask_m, ns, n_total, **kwargs)
    elif likelihood_family == 2:
        stats = lmmPoisson(Xm, ym, Zm, mask_n, mask_m, ns, n_total, **kwargs)
    else:
        raise ValueError(f'unsupported likelihood_family={likelihood_family}')

    if eta_rfx is not None:
        uncorr = (eta_rfx == 0)[:, None, None]  # (B, 1, 1)
        for key in ('Psi', 'Psi_pql', 'Psi_lap'):
            if key in stats:
                P = stats[key]
                stats[key] = torch.where(uncorr, torch.diag_embed(P.diagonal(dim1=-2, dim2=-1)), P)

    return stats
