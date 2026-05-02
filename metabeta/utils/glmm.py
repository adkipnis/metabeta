"""LMM/GLMM variance-component estimators: lmmNormal (Gaussian),
lmmBernoulli (logit), lmmPoisson (log). All return the same dict keys.
"""

import torch
import torch.nn.functional as F

from metabeta.utils.glm import _adaptiveRidge, _safeSolve, irlsBernoulliCompacted, irlsPoissonCompacted
from metabeta.utils.regularization import unconstrainedToCholesky


def _adaptiveRidgeBm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-adaptive ridge for (B, m, q, q) block matrices."""
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)  # (B, m)
    q = A.shape[-1]
    return eps * scale[..., None, None] * torch.eye(q, device=A.device, dtype=A.dtype)


_EIGH_JITTERS = [1e-6, 1e-4, 1e-2, 1.0]


def _eighWithJitter(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch eigh with fallback to element-wise jitter for ill-conditioned matrices.

    Fast path: full-batch eigh with no overhead.
    On failure: retry each element individually with increasing diagonal jitter.
    Elements that fail even at max jitter get vals=0, vecs=I (soft failure).
    """
    try:
        return torch.linalg.eigh(M)
    except torch.linalg.LinAlgError:
        pass

    B, q, _ = M.shape
    eye = torch.eye(q, device=M.device, dtype=M.dtype)
    all_vals, all_vecs = [], []
    for i in range(B):
        Mi = M[i]
        result = None
        for jitter in _EIGH_JITTERS:
            try:
                result = torch.linalg.eigh(Mi + jitter * eye)
                break
            except torch.linalg.LinAlgError:
                continue
        if result is None:
            result = (torch.zeros(q, device=M.device, dtype=M.dtype), eye)
        all_vals.append(result[0])
        all_vecs.append(result[1])
    return torch.stack(all_vals), torch.stack(all_vecs)


def _psdProject(M: torch.Tensor) -> torch.Tensor:
    """Project a symmetric (B, q, q) matrix onto the PSD cone."""
    M = 0.5 * (M + M.mT)
    vals, vecs = _eighWithJitter(M)
    return vecs @ torch.diag_embed(vals.clamp(min=0.0)) @ vecs.mT


def _pseudoInverse(M: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse of a PSD (B, q, q) matrix via eigendecomposition."""
    vals, vecs = _eighWithJitter(M)
    tol = vals.amax(dim=-1, keepdim=True).clamp(min=1e-8) * 1e-6
    inv_vals = torch.where(vals > tol, 1.0 / vals.clamp(min=1e-30), torch.zeros_like(vals))
    return vecs @ torch.diag_embed(inv_vals) @ vecs.mT


def _ridgeInv(M: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """Ridge-regularized inverse of a PSD (B, q, q) matrix.

    floor: (B,) per-batch scalar added to all eigenvalues before inversion.
    Unlike pseudoinverse, never zeros any eigenvalue — ensures strong shrinkage
    when M eigenvalues are near or below floor.
    """
    vals, vecs = _eighWithJitter(M)
    inv_vals = 1.0 / (vals + floor[:, None]).clamp(min=1e-30)
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


def _pqlPass(
    beta: torch.Tensor,      # (B, d) fixed effects for this pass
    Psi_inv: torch.Tensor,   # (B, q, q) precision used for Newton penalty and Hg
    Xm: torch.Tensor,        # (B, m, n, d)
    ym: torch.Tensor,        # (B, m, n)
    Zm: torch.Tensor,        # (B, m, n, q)
    mask_n: torch.Tensor,    # (B, m, n)
    mask_m: torch.Tensor,    # (B, m)
    mask4: torch.Tensor,     # (B, m, 1, 1)
    active: torch.Tensor,    # (B, m) bool
    eye_q: torch.Tensor,     # (q, q)
    eye_q_bm: torch.Tensor,  # (B, m, q, q)
    G: torch.Tensor,         # (B,)
    likelihood_family: int,
    n_newton: int = 3,
    bg_init: torch.Tensor | None = None,  # warm start; None = cold start from zeros
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        else:
            mu_t = torch.exp(eta_t.clamp(max=20))
        w_t = (mu_t * (1.0 - mu_t) if likelihood_family == 1 else mu_t).clamp(min=1e-6) * mask_n
        ZWZ_t = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_t, Zm)         # (B, m, q, q)
        grad_g = (
            torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_t) * mask_n)      # Zᵀ(y−μ)
            - torch.einsum('bqr,bmr->bmq', Psi_inv, bg)                   # −Ψ⁻¹b
        )
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
    bg_outer = torch.einsum('bmq,bmr->bmqr', bg, bg)
    Psi_lap = _psdProject((bg_outer + Hg_inv).sum(dim=1) / G[:, None, None])  # (B, q, q)
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
    A_g = XWX_f - torch.einsum(                                            # Schur complement
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
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
    resid_gls = (ytilde_f - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    blups = torch.einsum(
        'bmqr,bmr->bmq', Kg_inv,
        torch.einsum('bmnq,bmn->bmq', Zm, w_f * resid_gls),
    ) * mask_m[:, :, None]                                                 # (B, m, q)

    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    # Return Kg_inv for blup_var: (ZWZ + Psi_lap^{-1})^{-1} is the posterior covariance
    # of b_g under the final Psi_lap estimate — consistent with the Normal path's se²·W_g.
    return beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls


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

    Returns a dict with keys: beta_est, beta_wg, sigma_eps_est, sigma_rfx_est,
    blup_est, blup_var, bhat, resid_g, Psi.
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
    A_gls_reg = A_gls + _adaptiveRidge(A_gls)
    beta_gls = _safeSolve(A_gls_reg, b_gls)

    # clamp before BLUP computation: near-cancellation in A_gls=(XtX−XbarXbar) when λ→1
    # produces finite-but-huge values that nan_to_num cannot catch.
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
    sigma_eps = sigma_eps_sq.sqrt().unsqueeze(-1).nan_to_num(nan=1.0, posinf=1.0)  # (B, 1)
    sigma_rfx = sigma_rfx_sq.sqrt().unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)  # (B, 1)

    # BLUPs for q=1: EM-refined shrinkage.
    sigma_rfx_sq_val = sigma_rfx.squeeze(-1).square()              # (B,)  MoM, may be 0
    sigma_eps_sq_val = sigma_eps.squeeze(-1).square()              # (B,)

    r_g = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta_gls)) * mask_m

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
        A_gls_reg = A_gls + _adaptiveRidge(A_gls)
        beta_gls = _safeSolve(A_gls_reg, b_gls)
        beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        r_g = (y_mean - torch.einsum('bmd,bd->bm', X_mean, beta_gls)) * mask_m
        blups = (lambda_g2 * r_g).unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        blup_var = (sigma_rfx_sq_val[:, None] * (1.0 - lambda_g2)).clamp(min=0.0).unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)

    # Inflate blup_var to account for uncertainty in the sigma_rfx² estimate.
    # Var[sigma_rfx²] ~ 2*(sigma_rfx²)²/(G-d) (chi-squared); delta-method propagation
    # gives a multiplicative factor 1 + 2/(G-d), correcting the systematic overconfidence
    # that arises when treating estimated sigma_rfx as known.
    df_sigma = (G - d).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma)[:, None, None]

    # Kackar-Harville correction: blup_var conditions on β as known, but actual BLUP error
    # also includes λ_g² * x̄_g' Var(β_hat) x̄_g. Dominant for large groups where λ→1, (1-λ)Ψ→0.
    eye_d = torch.eye(d, device=Xm.device, dtype=Xm.dtype).expand(B, d, d)
    beta_var = _safeSolve(A_gls_reg, eye_d).diagonal(dim1=-1, dim2=-2).clamp(min=1e-8)  # (B, d)
    kh_corr = (lambda_g2.unsqueeze(-1) ** 2 * (X_mean ** 2 * beta_var[:, None, :]).sum(dim=-1, keepdim=True))  # (B, m, 1)
    blup_var = blup_var + kh_corr

    sigma_rfx = sigma_rfx_sq_val.clamp(min=0.0).sqrt().unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0)
    Psi = sigma_rfx.square().unsqueeze(-1)                          # (B, 1, 1)
    sigma_eps = sigma_eps_sq_val.clamp(min=0.0).sqrt().unsqueeze(-1).nan_to_num(nan=1.0, posinf=1.0)

    resid_g = r_g.unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # (B, m, 1)

    beta_wg_out = beta_wg.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    bhat_out = resid_bg.unsqueeze(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # (B, m, 1)

    return {
        'beta_est': beta_gls,           # (B, d)
        # 'beta_var': beta_var,           # (B, d)
        'beta_wg': beta_wg_out,         # (B, d)
        'sigma_eps_est': sigma_eps,     # (B, 1)
        'sigma_rfx_est': sigma_rfx,     # (B, 1)
        'blup_est': blups,              # (B, m, 1)
        'blup_var': blup_var,           # (B, m, 1)
        'bhat': bhat_out,               # (B, m, 1)
        'resid_g': resid_g,             # (B, m, 1)
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

    Returns a dict with keys: beta_est, beta_wg, sigma_eps_est, sigma_rfx_est,
    blup_est, blup_var, bhat, resid_g, Psi.
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
    bhat = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Ztr) * mask_m[:, :, None]  # (B, m, q) b̂_g

    # For the Psi MoM, exclude groups with n_g ≤ q+1: their ZtZ is near-singular,
    # so ZtZ_inv blows up and bhat can be orders of magnitude larger than the true b_g,
    # inflating Psi_raw catastrophically. Groups with n_g > q+1 have sufficient
    # within-group df for a reliable OLS estimate.
    mom_mask = mask_m * (ns > q + 1).float()    # (B, m)
    G_mom = mom_mask.sum(dim=1).clamp(min=1.0)  # (B,)
    mom4 = mom_mask[:, :, None, None]            # (B, m, 1, 1)

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat, bhat)        # (B, m, q, q)
    ZtZ_bhat = torch.einsum('bmqp,bmpk->bmqk', ZtZ_safe, bhat_outer)  # (B, m, q, q)

    mask4 = mask_m[:, :, None, None]                              # (B, m, 1, 1)
    sum_ZtZ = (ZtZ_safe * mom4).sum(dim=1)                        # (B, q, q) safe groups only
    sum_ZtZ_bhat = (ZtZ_bhat * mom4).sum(dim=1)                   # (B, q, q) safe groups only

    rhs_mom = sum_ZtZ_bhat - sigma_eps_sq[:, None, None] * G_mom[:, None, None] * eye_q
    Psi_raw = _safeSolve(sum_ZtZ + _adaptiveRidge(sum_ZtZ), rhs_mom)  # (B, q, q)

    # Per-component noise-corrected Psi diagonal floor to prevent EM getting stuck near Psi=0.
    # E[bhat_i²] ≈ Psi_ii + σ_ε² mean_g(ZtZ_inv_ii). Removing the noise term gives an
    # estimate of Psi_ii ≈ σ_rfx_i², which is near 0 when σ_rfx_i is small (so the floor
    # only activates for high-SNR components where MoM tends to under-estimate).
    # Applied per-component so inactive rfx dimensions (second Z column = 0 for q=1 datasets)
    # are not inflated — their signal_var ≈ 0 and the floor stays at 0.
    # Uses mom_mask to exclude near-singular groups (same as MoM sums above).
    mean_ZtZ_inv_diag = (
        ZtZ_inv.diagonal(dim1=-2, dim2=-1) * mom_mask[:, :, None]
    ).sum(dim=1) / G_mom[:, None]                                       # (B, q)
    bhat_var = (bhat.square() * mom_mask[:, :, None]).sum(dim=1) / G_mom[:, None]  # (B, q)
    psi_diag_floor = (bhat_var - sigma_eps_sq[:, None] * mean_ZtZ_inv_diag).clamp(min=0.0) * 0.5
    Psi_raw = Psi_raw + torch.diag_embed(
        (psi_diag_floor - Psi_raw.diagonal(dim1=-2, dim2=-1)).clamp(min=0.0)
    )                                                                   # bump diag to floor

    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    vals, vecs = _eighWithJitter(Psi_raw)                         # (B, q), (B, q, q)
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
    A_gls_reg = A_gls + _adaptiveRidge(A_gls)
    beta_gls = _safeSolve(A_gls_reg, b_gls)                        # (B, d)

    # clamp before BLUP computation: near-cancellation in A_gls=(XtX−correction_XX) when
    # Ψ≈0 produces finite-but-huge values that nan_to_num cannot catch.
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
        vals_r, vecs_r = _eighWithJitter(psi_reg)
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
        A_gls_reg = A_gls + _adaptiveRidge(A_gls)
        beta_gls = _safeSolve(A_gls_reg, b_gls)
        beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
        Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)

        # E-step: BLUPs and posterior variance
        blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        blup_var = (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
        blup_var = blup_var.nan_to_num(nan=0.0, posinf=0.0)

    # Inflate blup_var to account for uncertainty in the Psi estimate (same rationale
    # as the compacted path: Var[Psi] ∝ Psi²/(G-d), delta-method gives 1 + 2/(G-d)).
    df_sigma = (G - d).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma)[:, None, None]

    # Kackar-Harville correction: blup_var = diag(σ²W_g) conditions on β as known, but actual
    # BLUP error also includes W_g Z^T X (β_hat - β). Dominant for large groups where W_g→Ψ⁻¹,
    # (1-λ)Ψ→0. beta_var = diag(A_gls_reg⁻¹), W_ZtX = W_g Z^T X already computed in EM loop.
    eye_d_kh = torch.eye(d, device=Xm.device, dtype=Xm.dtype).expand(B, d, d)
    beta_var_kh = _safeSolve(A_gls_reg, eye_d_kh).diagonal(dim1=-1, dim2=-2).clamp(min=1e-8)  # (B, d)
    kh_corr = (W_ZtX ** 2 * beta_var_kh[:, None, None, :]).sum(dim=-1)  # (B, m, q)
    blup_var = blup_var + kh_corr

    # Floor blup_var at Psi_diag / (2 * n_g): prevents near-zero declared variance for
    # small groups on real (sampled) data where the Gaussian model may be misspecified.
    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)  # (B, q)
    blup_var_floor = psi_diag[:, None, :] / (2.0 * ns.clamp(min=1.0)[:, :, None])
    blup_var = blup_var.clamp(min=blup_var_floor)

    sigma_rfx = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()          # (B, q)
    sigma_eps_1d = se2.clamp(min=0.0).sqrt().nan_to_num(nan=1.0, posinf=1.0)  # (B,)

    ns_f_loc = ns.clamp(min=1.0)                                              # (B, m)
    resid_g = (resid_gls.sum(dim=2) / ns_f_loc * mask_m).unsqueeze(-1)       # (B, m, 1)
    resid_g = resid_g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    beta_wg_out = beta_wg.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    # Clamp bhat output: pathological groups (n_g ≈ q) still have bhat blown up via ZtZ_inv,
    # but they were excluded from MoM so Psi is clean. Cap output to ±10 for NN input safety.
    bhat_out = bhat.clamp(-10.0, 10.0).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # (B, m, q)

    return {
        'beta_est': beta_gls,                       # (B, d)
        'beta_var': beta_var_kh,                    # (B, d)
        'beta_wg': beta_wg_out,                     # (B, d)
        'sigma_eps_est': sigma_eps_1d.unsqueeze(-1), # (B, 1)
        'sigma_rfx_est': sigma_rfx,                 # (B, q)
        'blup_est': blups,                          # (B, m, q)
        'blup_var': blup_var,                       # (B, m, q)
        'bhat': bhat_out,                           # (B, m, q)
        'resid_g': resid_g,                         # (B, m, 1)
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
    vals_pql, vecs_pql = _eighWithJitter(Psi_pql)
    vals_pql = vals_pql.clamp(min=psi_0[:, None])
    Psi_pql = vecs_pql @ torch.diag_embed(vals_pql) @ vecs_pql.mT        # (B, q, q)

    # ------------------------------------------------------------------
    # Stage 2: alternating Newton–GLS loop, up to max_passes=6.
    # Pass 1 cold-starts Newton under pooled β₀ and pseudoinv(Ψ̂_PQL).
    # Passes 2–max_passes warm-start from previous BLUPs under β̂_GLS and
    # ridge-inv(Ψ̂_Lap); early exit when the 95th-percentile change in both
    # β and diag(Ψ̂) falls below 1e-3.
    # ------------------------------------------------------------------
    pass_args = (Xm, ym, Zm, mask_n, mask_m, mask4, active, eye_q, eye_q_bm, G, likelihood_family, n_newton)
    psi_0_floor = psi_0.clamp(min=1e-6)
    max_passes = 6

    def _sanitize(b_gls: torch.Tensor, psi: torch.Tensor):
        """Clamp beta_gls and clean Psi_lap before using as next-pass inputs."""
        return (
            b_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0),
            psi.nan_to_num(nan=0.0, posinf=0.0),
        )

    # Pass 1: pooled β₀, pseudoinverse of Psi_pql (cold start)
    Psi_inv = _pseudoInverse(Psi_pql)
    beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls = _pqlPass(
        beta_0, Psi_inv, *pass_args
    )
    beta_gls, Psi_lap = _sanitize(beta_gls, Psi_lap)

    # Passes 2–max_passes: warm start, ridge-regularized Ψ̂_Lap, until convergence.
    # Each pass refines (β, b̂_g, Ψ̂_Lap) jointly. Early exit when the 95th-percentile
    # change across the batch is below tolerance — avoids running all max_passes when a
    # few pathological datasets (small m) never satisfy the strict amax criterion.
    for _ in range(max_passes - 1):
        beta_prev = beta_gls
        psi_diag_prev = Psi_lap.diagonal(dim1=-2, dim2=-1)

        Psi_inv = _ridgeInv(Psi_lap, psi_0_floor)
        beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls = _pqlPass(
            beta_gls, Psi_inv, *pass_args, bg_init=blups
        )
        beta_gls, Psi_lap = _sanitize(beta_gls, Psi_lap)

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

    # Per-group mean working residual (after removing fixed effects)
    resid_g = (resid_gls.sum(dim=2) / ns.clamp(min=1.0) * mask_m).unsqueeze(-1)  # (B, m, 1)
    resid_g = resid_g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    beta_wg_out = beta_0.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # pooled IRLS (no rfx)
    bhat_out = bhat_ols.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)   # (B, m, q)

    return {
        'beta_est': beta_gls,           # (B, d)
        'beta_wg': beta_wg_out,         # (B, d)  pooled IRLS (no-rfx analog of beta_wg)
        'sigma_rfx_est': sigma_rfx_est, # (B, q)
        'blup_est': blups,              # (B, m, q)
        'blup_var': blup_var,           # (B, m, q)
        'bhat': bhat_out,               # (B, m, q)
        'resid_g': resid_g,             # (B, m, 1)
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


# ---------------------------------------------------------------------------
# Analytical BLUP context for the local-flow conditioning signal
# ---------------------------------------------------------------------------


def analyticalBLUPContext(
    data: dict[str, torch.Tensor],
    beta: torch.Tensor,           # (B, S, d) fixed effects — constrained scale
    sigma_rfx: torch.Tensor,      # (B, S, q) — constrained scale
    sigma_eps: torch.Tensor,      # (B, S)   — constrained scale
    z_corr: torch.Tensor | None,  # (B, S, d_corr) unconstrained atanh or None
    clamp: float = 20.0,
) -> torch.Tensor:
    """Analytical BLUP mean, marginal std, and shrinkage given global parameter samples.

    Uses the closed-form Gaussian posterior (family == 0 only).
    Returns (B, m, S, 3*q).
    """
    from metabeta.posthoc.gaussian_local import analyticalBLUPStats

    q = sigma_rfx.shape[-1]
    d = beta.shape[-1]

    Sigma_rfx_inv: torch.Tensor | None = None
    if z_corr is not None:
        L_corr = unconstrainedToCholesky(z_corr, q)                    # (B, S, q, q)
        sr_inv_diag = torch.diag_embed(1.0 / sigma_rfx.clamp(min=1e-6))  # (B, S, q, q)
        A = torch.linalg.solve_triangular(L_corr, sr_inv_diag, upper=False)
        Sigma_rfx_inv = A.mT @ A                                        # (B, S, q, q)

    mu, blup_std, lambda_g = analyticalBLUPStats(
        data['y'],
        data['X'][..., :d],
        data['Z'][..., :q],
        beta,
        sigma_rfx,
        sigma_eps,
        data['mask_n'],
        Sigma_rfx_inv=Sigma_rfx_inv,
    )
    return torch.cat([mu.clamp(-clamp, clamp), blup_std.clamp(max=clamp), lambda_g], dim=-1)
