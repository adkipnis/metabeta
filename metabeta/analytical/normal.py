"""Gaussian LMM analytical estimator."""

import torch

from metabeta.analytical.glm import _adaptiveRidge, _safeSolve
from metabeta.analytical.constants import _NORMAL_FULL_MIN_EM
from metabeta.analytical.linalg import (
    _adaptiveRidgeBm,
    _eighWithJitter,
    _gramRank,
    _groupZDiagnostics,
    _maskedMean,
    _maskedMedian,
    _psdClampEigenvalues,
    _psdProject,
)


def _lmmNormalFull(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,  # (B, m)     1 for active groups
    ns: torch.Tensor,  # (B, m)     group sizes (float, ≥ 1 for active)
    n_total: torch.Tensor,  # (B,)       total active observations
    n_em: int = 3,
    uncorr: torch.Tensor | None = None,  # (B,) bool — force Ψ diagonal for these datasets
    mask_q: torch.Tensor | None = None,  # (B, q) bool — active random-effect components
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
    if mask_q is not None:
        Zm = Zm * mask_q.to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :q]
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
    mom_mask, z_rank, _, active_components, active_count = _groupZDiagnostics(ZtZ, mask_m, ns, q)
    G_mom_raw = mom_mask.sum(dim=1)                               # (B,)
    G_mom = G_mom_raw.clamp(min=1.0)                              # (B,)
    mom4 = mom_mask[:, :, None, None]                              # (B, m, 1, 1)
    enough_full_mom = G_mom_raw >= torch.maximum(
        active_count + 1.0, G_mom_raw.new_full((B,), float(d + 1))
    )
    enough_diag_mom = (G_mom_raw >= 2.0) & (active_count > 0)
    active_q = active_components.to(Zm.dtype)
    active_qq = active_q[:, :, None] * active_q[:, None, :]

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
    mx_rank = _gramRank(MXtMX)
    df_w = (N - z_rank.sum(dim=1) - mx_rank).clamp(min=1.0)       # (B,)
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
    mean_ZtZ_inv_diag = (ZtZ_inv.diagonal(dim1=-2, dim2=-1) * mom_mask[:, :, None]).sum(
        dim=1
    ) / G_mom[
        :, None
    ]                                       # (B, q)
    # Center bhat over informative groups before squaring so that a shared beta_ols offset
    # (bhat_g ≈ b_g + (beta_true − beta_ols)) cancels in the variance rather than inflating
    # signal_mean and psi_eig_cap. Without centering, the squared offset can be O(1) while
    # sigma_rfx² ≈ 0.2, loosening the cap by 10–20× and producing catastrophic EM spikes.
    bhat_mean = _maskedMean(bhat, mom_mask[:, :, None], dim=1)  # (B, q)
    bhat_centered = bhat - bhat_mean[:, None, :]                 # (B, m, q)
    bhat_signal = (
        bhat_centered.square() - sigma_eps_sq[:, None, None] * ZtZ_inv.diagonal(dim1=-2, dim2=-1)
    ).clamp(min=0.0)
    signal_mean = _maskedMean(bhat_signal, mom_mask[:, :, None], dim=1)
    signal_cap = (6.0 * signal_mean).clamp(min=sigma_eps_sq[:, None] * 1e-6)
    signal_winsor = torch.minimum(bhat_signal, signal_cap[:, None, :])
    signal_mean = _maskedMean(signal_winsor, mom_mask[:, :, None], dim=1)
    signal_median = _maskedMedian(signal_winsor, mom_mask[:, :, None], dim=1)
    psi_diag_signal = torch.minimum(signal_mean, 2.0 * signal_median)

    active_ns_mean = _maskedMean(ns, mask_m, dim=1).clamp(min=1.0)
    fallback_diag = (sigma_eps_sq / active_ns_mean).clamp(min=1e-10)[:, None].expand(B, q)
    fallback_diag = fallback_diag * active_q
    psi_diag_floor = torch.where(
        enough_diag_mom[:, None],
        torch.maximum(0.5 * psi_diag_signal * active_q, fallback_diag),
        fallback_diag,
    )
    psi_eig_cap = torch.maximum(
        (6.0 * psi_diag_signal * active_q).amax(dim=1),
        fallback_diag.amax(dim=1),
    )
    Psi_raw = Psi_raw + torch.diag_embed(
        (psi_diag_floor - Psi_raw.diagonal(dim1=-2, dim2=-1)).clamp(min=0.0)
    )                                                                   # bump diag to floor

    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    Psi_raw = Psi_raw * active_qq
    Psi_raw = torch.where(enough_full_mom[:, None, None], Psi_raw, torch.diag_embed(psi_diag_floor))
    vals, vecs = _eighWithJitter(Psi_raw)                         # (B, q), (B, q, q)
    vals = vals.clamp(min=0.0)
    Psi = vecs @ torch.diag_embed(vals) @ vecs.mT                 # (B, q, q)
    Psi = _psdClampEigenvalues(Psi, psi_eig_cap)

    if uncorr is not None:
        Psi = torch.where(
            uncorr[:, None, None], torch.diag_embed(Psi.diagonal(dim1=-2, dim2=-1)), Psi
        )
        vals, vecs = _eighWithJitter(Psi)

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

    xtx_max_diag = XtX.diagonal(dim1=-1, dim2=-2).amax(dim=-1).clamp(min=1.0)  # (B,)
    # Per-component GLS identification: component k has sufficient between-group
    # information when (XtX − correction_XX)[k,k] is large relative to XtX scale.
    # A scalar max check would let unidentified components (near-zero diagonal) remain
    # in the GLS solve and blow up via the adaptive ridge, even when other components
    # are well-identified.  Per-component check handles each direction independently.
    comp_id = (XtX - correction_XX).diagonal(  # (B, d) bool
        dim1=-2, dim2=-1
    ).abs() > 1e-3 * xtx_max_diag[:, None]
    beta_identified = comp_id.any(dim=-1)                             # (B,) for KH/gls_determined

    A_gls_reg = A_gls + _adaptiveRidge(A_gls)
    beta_gls = _safeSolve(A_gls_reg, b_gls).nan_to_num(0.0, 0.0, 0.0)  # (B, d)
    # Fallback for GLS-unidentified components (correction_XX ≈ XtX along that direction):
    # use beta_wg when within-group X has variation (mx_rank > 0); use beta_ols (global OLS)
    # when MX = 0, i.e. X lies entirely in Z's column space (e.g. intercept-only model).
    # At very high SNR, GLS degenerates to OLS, so beta_ols is the correct limiting value.
    beta_gls_fallback = torch.where(mx_rank[:, None] > 0, beta_wg, beta_ols)
    beta_gls = torch.where(comp_id, beta_gls, beta_gls_fallback)
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
    beta_rank = mx_rank.clamp(min=1.0, max=float(d))
    for _ in range(n_em):
        # M-step: Ψ using full posterior covariance (exact for Gaussian b_g)
        blup_outer = torch.einsum('bmq,bmr->bmqr', blups, blups)     # (B, m, q, q)
        post_cov = se2[:, None, None, None] * W_g                    # (B, m, q, q)  σ²_ε W_g
        Psi_em = _psdProject(
            ((blup_outer + post_cov) * mom4).sum(dim=1) / G_mom[:, None, None]
        )  # (B, q, q)
        psi_diag_em = Psi_em.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)
        Psi_em = torch.where(enough_full_mom[:, None, None], Psi_em, torch.diag_embed(psi_diag_em))
        Psi = _psdClampEigenvalues(_psdProject((0.5 * Psi + 0.5 * Psi_em) * active_qq), psi_eig_cap)
        if uncorr is not None:
            Psi = torch.where(
                uncorr[:, None, None], torch.diag_embed(Psi.diagonal(dim1=-2, dim2=-1)), Psi
            )

        # M-step: σ_ε² (REML-like df correction using current blups and beta_gls)
        resid_em = (
            ym
            - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)
            - torch.einsum('bmnq,bmq->bmn', Zm, blups)
        ) * mask_n
        ss_em = resid_em.square().sum(dim=(1, 2))                    # (B,)
        ZtZ_W = torch.einsum('bmqr,bmrs->bmqs', ZtZ_safe, W_g)      # (B, m, q, q)
        T = (ZtZ_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1) * mask_m).sum(dim=1)  # (B,)
        # Cap T so the REML denominator N−d−T stays ≥ 10% of N−d, preventing blow-up
        # when T ≈ N−d (λ_g → 1 for all groups at high SNR or near-singular ZtZ_g).
        T_safe = T.clamp(max=0.9 * (N - beta_rank).clamp(min=1.0))
        se2 = (ss_em / (N - beta_rank - T_safe).clamp(min=1.0)).clamp(min=1e-12)

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
        beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        beta_gls = torch.where(beta_identified[:, None], beta_gls, beta_wg)
        resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
        Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)

        # E-step: BLUPs and posterior variance
        blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        blup_var = (
            (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
        )
        blup_var = blup_var.nan_to_num(nan=0.0, posinf=0.0)

    # Inflate blup_var to account for uncertainty in the Psi estimate.
    # Var[Psi] ∝ Psi²/(G-d), delta-method gives 1 + 2/(G-d).
    # Clamp denominator at 4 to cap inflation at 50% for large G; uncapped it over-inflates
    # blup_var in real-data regimes where Psi is well-estimated (observed ratio < 0.5).
    df_sigma = (G - d).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma.clamp(min=4.0))[:, None, None]

    # Kackar-Harville correction: blup_var = diag(σ²W_g) conditions on β as known, but actual
    # BLUP error also includes W_g Z^T X (β_hat - β). Dominant for large groups where W_g→Ψ⁻¹,
    # (1-λ)Ψ→0.
    # beta_var uses a truncated eigendecomposition of A_gls_reg: near-zero eigenvalues (rank
    # deficiency from m<d, collinear group means, or q≥d collapse) inflate beta_var_kh via
    # the adaptive-ridge reciprocal ≈1e6. Truncating below 1e-3 × max_eig caps beta_var at
    # the identified directions only, zeroing contributions from the null space.
    vals_kh, vecs_kh = torch.linalg.eigh(A_gls_reg)                # (B, d), (B, d, d)
    max_kh = vals_kh.amax(dim=-1, keepdim=True).clamp(min=1.0)     # (B, 1)
    inv_vals_kh = torch.where(
        vals_kh > 1e-3 * max_kh,
        1.0 / vals_kh.clamp(min=1e-30),
        torch.zeros_like(vals_kh),
    )
    beta_var_kh = (vecs_kh**2 * inv_vals_kh[:, None, :]).sum(dim=-1).clamp(min=1e-8)  # (B, d)
    kh_corr = (W_ZtX**2 * beta_var_kh[:, None, None, :]).sum(dim=-1)  # (B, m, q)
    # Zero KH for collapsed GLS (q≥d or G<d): those datasets use beta_wg so β-uncertainty
    # is not meaningful; truncation above handles the collinear-group-means case.
    gls_determined = G >= float(d)  # (B,)
    kh_corr = kh_corr * (beta_identified & gls_determined)[:, None, None]
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
        'beta_est': beta_gls,  # (B, d)
        'beta_var': beta_var_kh,  # (B, d)
        'beta_wg': beta_wg_out,  # (B, d)
        'sigma_eps_est': sigma_eps_1d.unsqueeze(-1),  # (B, 1)
        'sigma_rfx_est': sigma_rfx,  # (B, q)
        'blup_est': blups,  # (B, m, q)
        'blup_var': blup_var,  # (B, m, q)
        'bhat': bhat_out,  # (B, m, q)
        'resid_g': resid_g,  # (B, m, 1)
        'Psi': Psi,  # (B, q, q)
    }


def lmmNormal(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)
    mask_m: torch.Tensor,  # (B, m)
    ns: torch.Tensor,  # (B, m)
    n_total: torch.Tensor,  # (B,)
    n_em: int = 3,
    uncorr: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the Gaussian LMM."""
    return _lmmNormalFull(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        n_em=max(n_em, _NORMAL_FULL_MIN_EM),
        uncorr=uncorr,
        mask_q=mask_q,
    )
