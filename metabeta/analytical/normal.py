"""Gaussian LMM analytical estimator."""

from dataclasses import dataclass

import torch

from metabeta.analytical.constants import _NORMAL_FULL_MIN_EM
from metabeta.analytical.linalg import (
    _adaptiveRidge,
    _adaptiveRidgeBm,
    _eighWithJitter,
    _gramRank,
    _groupZDiagnostics,
    _maskedMean,
    _maskedMedian,
    _psdClampEigenvalues,
    _psdProject,
    _safeSolve,
    _shrinkOffDiagonal,
)


@dataclass(frozen=True)
class _NormalGlsResult:
    beta: torch.Tensor
    beta_mask: torch.Tensor
    W_g: torch.Tensor
    W_ZtX: torch.Tensor
    A_reg: torch.Tensor
    resid: torch.Tensor
    blups: torch.Tensor
    blup_var: torch.Tensor


def _forceDiagonalPsi(Psi: torch.Tensor, uncorr: torch.Tensor | None) -> torch.Tensor:
    if uncorr is None:
        return Psi
    return torch.where(uncorr[:, None, None], torch.diag_embed(Psi.diagonal(dim1=-2, dim2=-1)), Psi)


def _estimateWithinGroupVariance(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask_n: torch.Tensor,
    n_total: torch.Tensor,
    Zm: torch.Tensor,
    ZtZ_inv: torch.Tensor,
    ZtX: torch.Tensor,
    z_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate fixed effects and residual variance after projecting out Z."""
    d = Xm.shape[-1]
    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)
    Zhat_y = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Zty)
    Zhat_X = torch.einsum('bmqr,bmrd->bmqd', ZtZ_inv, ZtX)
    My = (ym - torch.einsum('bmnq,bmq->bmn', Zm, Zhat_y)) * mask_n
    MX = (Xm - torch.einsum('bmnq,bmqd->bmnd', Zm, Zhat_X)) * mask_n[..., None]

    MXtMX = torch.einsum('bmnd,bmnk->bdk', MX, MX)
    MXtMy = torch.einsum('bmnd,bmn->bd', MX, My)
    beta_wg = _safeSolve(MXtMX + _adaptiveRidge(MXtMX), MXtMy)

    resid_M = My - torch.einsum('bmnd,bd->bmn', MX, beta_wg)
    mx_rank = _gramRank(MXtMX)
    df_w = (n_total.float() - z_rank.sum(dim=1) - mx_rank).clamp(min=1.0)
    sigma_eps_sq = (resid_M.square().sum(dim=(1, 2)) / df_w).clamp(min=0.0)
    return beta_wg, sigma_eps_sq, mx_rank


def _componentwisePsiDiagSignal(
    Zm: torch.Tensor,
    resid: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    sigma_eps_sq: torch.Tensor,
    active_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate diagonal Psi signal one component at a time.

    The full MoM path requires groups whose Z design identifies all active random-effect
    components jointly. High-q sampled datasets often have no such groups, but individual
    components can still be estimated from groups where that column has support.
    """
    z2 = torch.einsum('bmnq,bmnq->bmq', Zm, Zm)
    ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    z2_sum = (z2 * mask_m[:, :, None]).sum(dim=1)
    z2_tol = torch.maximum(z2_sum[:, None, :] * 1e-6, z2.new_tensor(1e-8))
    component_mask = (
        mask_m[:, :, None].bool()
        & active_q[:, None, :].bool()
        & (ns[:, :, None] > 2.0)
        & (z2 > z2_tol)
    ).to(Zm.dtype)
    component_count = component_mask.sum(dim=1)

    bhat = ztr / z2.clamp(min=1e-8)
    bhat_mean = _maskedMean(bhat, component_mask, dim=1)
    bhat_centered = bhat - bhat_mean[:, None, :]
    bhat_noise = sigma_eps_sq[:, None, None] / z2.clamp(min=1e-8)
    bhat_signal = (bhat_centered.square() - bhat_noise).clamp(min=0.0)
    signal_mean = _maskedMean(bhat_signal, component_mask, dim=1)
    signal_cap = (6.0 * signal_mean).clamp(min=sigma_eps_sq[:, None] * 1e-6)
    signal_winsor = torch.minimum(bhat_signal, signal_cap[:, None, :])
    signal_mean = _maskedMean(signal_winsor, component_mask, dim=1)
    signal_median = _maskedMedian(signal_winsor, component_mask, dim=1)
    return torch.minimum(signal_mean, 2.0 * signal_median) * active_q, component_count


def _initialPsiMom(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    ZtZ_safe: torch.Tensor,
    ZtZ_inv: torch.Tensor,
    XtX: torch.Tensor,
    Xty: torch.Tensor,
    sigma_eps_sq: torch.Tensor,
    mom_mask: torch.Tensor,
    G_mom: torch.Tensor,
    enough_full_mom: torch.Tensor,
    enough_diag_mom: torch.Tensor,
    active_q: torch.Tensor,
    active_qq: torch.Tensor,
    eye_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initial method-of-moments estimate for Psi and group OLS effects."""
    B, q = Xm.shape[0], Zm.shape[-1]
    beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
    resid_full = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_ols)) * mask_n
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid_full)
    bhat = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Ztr) * mask_m[:, :, None]

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat, bhat)
    ZtZ_bhat = torch.einsum('bmqp,bmpk->bmqk', ZtZ_safe, bhat_outer)
    mom4 = mom_mask[:, :, None, None]
    sum_ZtZ = (ZtZ_safe * mom4).sum(dim=1)
    sum_ZtZ_bhat = (ZtZ_bhat * mom4).sum(dim=1)

    rhs_mom = sum_ZtZ_bhat - sigma_eps_sq[:, None, None] * G_mom[:, None, None] * eye_q
    Psi_raw = _safeSolve(sum_ZtZ + _adaptiveRidge(sum_ZtZ), rhs_mom)

    bhat_mean = _maskedMean(bhat, mom_mask[:, :, None], dim=1)
    bhat_centered = bhat - bhat_mean[:, None, :]
    bhat_signal = (
        bhat_centered.square() - sigma_eps_sq[:, None, None] * ZtZ_inv.diagonal(dim1=-2, dim2=-1)
    ).clamp(min=0.0)
    signal_mean = _maskedMean(bhat_signal, mom_mask[:, :, None], dim=1)
    signal_cap = (6.0 * signal_mean).clamp(min=sigma_eps_sq[:, None] * 1e-6)
    signal_winsor = torch.minimum(bhat_signal, signal_cap[:, None, :])
    signal_mean = _maskedMean(signal_winsor, mom_mask[:, :, None], dim=1)
    signal_median = _maskedMedian(signal_winsor, mom_mask[:, :, None], dim=1)
    psi_diag_signal = torch.minimum(signal_mean, 2.0 * signal_median)
    component_diag_signal, component_count = _componentwisePsiDiagSignal(
        Zm, resid_full, mask_m, ns, sigma_eps_sq, active_q
    )
    has_joint_mom = mom_mask.sum(dim=1) > 0
    reliable_component = component_count >= 5.0
    use_component_diag = reliable_component & ~has_joint_mom[:, None]

    active_ns_mean = _maskedMean(ns, mask_m, dim=1).clamp(min=1.0)
    fallback_diag = (sigma_eps_sq / active_ns_mean).clamp(min=1e-10)[:, None].expand(B, q)
    fallback_diag = fallback_diag * active_q
    diag_floor_signal = torch.where(
        enough_diag_mom[:, None],
        0.45 * psi_diag_signal * active_q,
        0.1 * component_diag_signal * active_q,
    )
    enough_diag = enough_diag_mom[:, None] | use_component_diag
    psi_diag_floor = torch.where(
        enough_diag,
        torch.maximum(diag_floor_signal, fallback_diag),
        fallback_diag,
    )
    # Gate cap signal per-component: unreliable components (< 5 valid groups) should not
    # inflate psi_eig_cap via amax — fall back to the conservative fallback_diag signal.
    diag_cap_signal = torch.where(
        enough_diag_mom[:, None],
        psi_diag_signal,
        torch.where(reliable_component, component_diag_signal, fallback_diag),
    )
    psi_eig_cap = torch.maximum(
        (6.0 * diag_cap_signal * active_q).amax(dim=1),
        fallback_diag.amax(dim=1),
    )
    Psi_raw = Psi_raw + torch.diag_embed(
        (psi_diag_floor - Psi_raw.diagonal(dim1=-2, dim2=-1)).clamp(min=0.0)
    )

    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    Psi_raw = Psi_raw * active_qq
    Psi_raw = torch.where(enough_full_mom[:, None, None], Psi_raw, torch.diag_embed(psi_diag_floor))
    Psi = _psdClampEigenvalues(_psdProject(Psi_raw), psi_eig_cap)
    return beta_ols, bhat, Psi, psi_diag_floor, psi_eig_cap


def _normalGlsAndBlups(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    ZtZ_safe: torch.Tensor,
    Zty: torch.Tensor,
    ZtX: torch.Tensor,
    XtX: torch.Tensor,
    Xty: torch.Tensor,
    XtZ: torch.Tensor,
    Psi: torch.Tensor,
    se2: torch.Tensor,
    eye_q: torch.Tensor,
    eye_q_bm: torch.Tensor,
    mask4: torch.Tensor,
    beta_fallback: torch.Tensor,
    beta_mask: torch.Tensor | None = None,
) -> _NormalGlsResult:
    """One Gaussian GLS/BLUP step for fixed Psi and residual variance."""
    vals, vecs = _eighWithJitter(Psi + se2[:, None, None] * 1e-4 * eye_q)
    Psi_inv = vecs @ torch.diag_embed(1.0 / vals.clamp(min=1e-30)) @ vecs.mT

    inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm) * mask4

    W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)
    correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)
    W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)
    correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)

    inv_se2 = 1.0 / se2
    A_gls = inv_se2[:, None, None] * (XtX - correction_XX)
    b_gls = inv_se2[:, None] * (Xty - correction_Xy)
    A_reg = A_gls + _adaptiveRidge(A_gls)
    beta_gls = _safeSolve(A_reg, b_gls).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    if beta_mask is None:
        xtx_max_diag = XtX.diagonal(dim1=-1, dim2=-2).amax(dim=-1).clamp(min=1.0)
        beta_mask = (XtX - correction_XX).diagonal(dim1=-2, dim2=-1).abs() > (
            1e-3 * xtx_max_diag[:, None]
        )
    beta_gls = torch.where(beta_mask, beta_gls, beta_fallback)

    resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)
    blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    blup_var = (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1)
    blup_var = blup_var.clamp(min=0.0, max=25.0).nan_to_num(nan=0.0, posinf=0.0)

    return _NormalGlsResult(
        beta=beta_gls,
        beta_mask=beta_mask,
        W_g=W_g,
        W_ZtX=W_ZtX,
        A_reg=A_reg,
        resid=resid_gls,
        blups=blups,
        blup_var=blup_var,
    )


def _remlNewtonStep(
    Zm: torch.Tensor,
    gls: _NormalGlsResult,
    ZtZ_safe: torch.Tensor,
    se2: torch.Tensor,
    Psi: torch.Tensor,
    mask_m: torch.Tensor,
    active_qq: torch.Tensor,
    psi_diag_floor: torch.Tensor,
    psi_eig_cap: torch.Tensor,
    uncorr: torch.Tensor | None,
) -> torch.Tensor:
    """One diagonal REML-Newton step for variance components ψ_k = diag(Ψ).

    Score: S_k = -½ Σ_g H_{g,kk} + ½ Σ_g (Z_g'V_g⁻¹r_g)²_k, with the exact
    Z'V⁻¹r from Woodbury. Expected Fisher info: F_k = ½ Σ_g H_{g,kk}². Damped
    update: Δψ_k = 0.5·S_k/F_k. The exact signal is ~0 at the MLE (safe for all
    datasets without gating) and strictly positive at Ψ=0 (escapes the EM trap).
    """
    active = mask_m.bool()  # (B, m)
    se2_4d = se2[:, None, None, None]  # (B, 1, 1, 1) for 4D matrix ops
    se2_3d = se2[:, None, None]        # (B, 1, 1) for 3D ops

    # Diagonal of H_g = Z'V_g⁻¹Z per group via Woodbury: V_g⁻¹ = σ⁻²(I - Z W Z'/σ²)
    ZtZ_W = torch.einsum('bmqr,bmrs->bmqs', ZtZ_safe, gls.W_g)
    ZtZ_W_ZtZ = torch.einsum('bmqr,bmrs->bmqs', ZtZ_W, ZtZ_safe)
    H_diag = (ZtZ_safe - ZtZ_W_ZtZ / se2_4d).diagonal(dim1=-2, dim2=-1) / se2_3d
    H_diag = H_diag.clamp(min=0.0) * active[:, :, None].to(H_diag.dtype)  # (B, m, q)

    # Exact REML signal: (Z'V_g⁻¹r_g)²_k via Woodbury V_g⁻¹=σ⁻²(I-Z W Z'/σ²)
    # Z'V⁻¹r = (Ztr - ZtZ_W @ Ztr / σ²) / σ². At Ψ=0: W_g→0, reduces to Ztr/σ².
    # At true Ψ: gradient→0, so this is safe without a gate (no overshoot at MLE).
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, gls.resid)  # (B, m, q)
    ZtZ_W_Ztr = torch.einsum('bmqr,bmr->bmq', ZtZ_W, Ztr)  # (B, m, q)
    Vr = (Ztr - ZtZ_W_Ztr / se2_3d) / se2_3d  # Z'V_g⁻¹r per group
    signal = Vr.square() * active[:, :, None].to(Vr.dtype)

    S = -0.5 * H_diag.sum(dim=1) + 0.5 * signal.sum(dim=1)  # (B, q)
    F = 0.5 * H_diag.square().sum(dim=1).clamp(min=1e-10)    # (B, q)

    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)
    psi_diag_new = (psi_diag + 0.5 * S / F).clamp(min=psi_diag_floor)

    return _psdClampEigenvalues(
        _forceDiagonalPsi(torch.diag_embed(psi_diag_new) * active_qq, uncorr),
        psi_eig_cap,
    )


def _emRefineNormal(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    ZtZ_safe: torch.Tensor,
    Zty: torch.Tensor,
    ZtX: torch.Tensor,
    XtX: torch.Tensor,
    Xty: torch.Tensor,
    XtZ: torch.Tensor,
    eye_q: torch.Tensor,
    eye_q_bm: torch.Tensor,
    mask4: torch.Tensor,
    mom4: torch.Tensor,
    G_mom: torch.Tensor,
    enough_full_mom: torch.Tensor,
    active_qq: torch.Tensor,
    beta_wg: torch.Tensor,
    beta_rank: torch.Tensor,
    beta_mask: torch.Tensor,
    Psi: torch.Tensor,
    se2: torch.Tensor,
    psi_diag_floor: torch.Tensor,
    psi_eig_cap: torch.Tensor,
    gls: _NormalGlsResult,
    n_em: int,
    uncorr: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, _NormalGlsResult]:
    """EM refinement of Psi, sigma_eps_sq, beta, and BLUPs."""
    N = n_total.float()
    se2_anchor = se2
    for _ in range(n_em):
        psi_prev = Psi
        se2_prev = se2

        # Winsorize BLUPs before M-step outer product: prevents groups with numerically
        # blown BLUPs (GLS near-singularity at high d/q) from spiking Ψ_em.
        # Cap at 10×σ̂_rfx using the current Ψ so the threshold adapts as Ψ converges.
        psi_diag_cur = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor).sqrt()
        blup_cap = 10.0 * psi_diag_cur[:, None, :]  # (B, 1, q) broadcasts over groups
        blups_winsor = gls.blups.clamp(min=-blup_cap, max=blup_cap)
        blup_outer = torch.einsum('bmq,bmr->bmqr', blups_winsor, blups_winsor)
        post_cov = se2[:, None, None, None] * gls.W_g
        # Outlier-trimmed Ψ_em: exclude groups whose ||b̂_g||² exceeds 3× the mean
        # among mom groups. This is adaptive — when all groups have similar norms (no
        # outliers) no groups are excluded and Ψ_em equals the original full mean.
        # Only kicks in when one or more groups are genuinely anomalous relative to the
        # rest, which is the case where Fix 9's per-BLUP cap occasionally over-clips.
        blup_norm = blups_winsor.square().sum(dim=-1)  # (B, m)
        mom_mask_1d = mom4.squeeze(-1).squeeze(-1).bool()  # (B, m)
        mom_norm_mean = (blup_norm * mom_mask_1d.float()).sum(dim=1) / G_mom  # (B,)
        outlier_thresh = 3.0 * mom_norm_mean[:, None]  # (B, 1) — 3× mean as outlier gate
        trim_mask = mom_mask_1d & (blup_norm <= outlier_thresh)  # (B, m) True = included
        trim_count = trim_mask.float().sum(dim=1).clamp(min=1.0)  # (B,)
        trim_mask_4d = trim_mask[:, :, None, None]
        Psi_em = _psdProject(
            ((blup_outer + post_cov) * trim_mask_4d).sum(dim=1) / trim_count[:, None, None]
        )
        psi_diag_em = Psi_em.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)
        Psi_em = torch.where(enough_full_mom[:, None, None], Psi_em, torch.diag_embed(psi_diag_em))
        Psi = _psdClampEigenvalues(_psdProject((0.5 * Psi + 0.5 * Psi_em) * active_qq), psi_eig_cap)
        Psi = _forceDiagonalPsi(Psi, uncorr)

        resid_em = (
            ym
            - torch.einsum('bmnd,bd->bmn', Xm, gls.beta)
            - torch.einsum('bmnq,bmq->bmn', Zm, gls.blups)
        ) * mask_n
        ss_em = resid_em.square().sum(dim=(1, 2))
        ZtZ_W = torch.einsum('bmqr,bmrs->bmqs', ZtZ_safe, gls.W_g)
        T = (ZtZ_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1) * mask_m).sum(dim=1)
        T_safe = T.clamp(max=0.9 * (N - beta_rank).clamp(min=1.0))
        se2_next = (ss_em / (N - beta_rank - T_safe).clamp(min=1.0)).clamp(min=1e-12)
        # In weakly identified batches, unstable beta/BLUP passes can repeatedly inflate
        # residual variance and poison the next GLS step. Large genuine changes should
        # already be visible in the projection estimate, so cap EM growth relative to it.
        se2 = torch.minimum(se2_next, (100.0 * se2_anchor).clamp(min=1e-12))

        gls = _normalGlsAndBlups(
            Xm,
            ym,
            Zm,
            mask_n,
            ZtZ_safe,
            Zty,
            ZtX,
            XtX,
            Xty,
            XtZ,
            Psi,
            se2,
            eye_q,
            eye_q_bm,
            mask4,
            beta_wg,
            beta_mask,
        )

        psi_delta = (Psi - psi_prev).abs().amax(dim=(-2, -1)).amax()
        se2_delta = (se2 - se2_prev).abs().amax()
        if psi_delta < 1e-3 and se2_delta < 1e-3:
            break

    return Psi, se2, gls


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
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    if mask_q is not None:
        Zm = Zm * mask_q.to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :q]
    N = n_total.float()                               # (B,)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,)
    active = mask_m.bool()                            # (B, m)
    mask4 = mask_m[:, :, None, None]                  # (B, m, 1, 1)

    eye_q = torch.eye(q, device=Xm.device, dtype=Xm.dtype)       # (q, q)
    eye_q_bm = eye_q.expand(B, m, q, q)                          # (B, m, q, q)

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

    beta_wg, sigma_eps_sq, mx_rank = _estimateWithinGroupVariance(
        Xm, ym, mask_n, n_total, Zm, ZtZ_inv, ZtX, z_rank
    )
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)                 # (B, d, d)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)                   # (B, d)
    beta_ols, bhat, Psi, psi_diag_floor, psi_eig_cap = _initialPsiMom(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        ZtZ_safe,
        ZtZ_inv,
        XtX,
        Xty,
        sigma_eps_sq,
        mom_mask,
        G_mom,
        enough_full_mom,
        enough_diag_mom,
        active_q,
        active_qq,
        eye_q,
    )
    Psi = _forceDiagonalPsi(Psi, uncorr).nan_to_num(nan=0.0, posinf=0.0)
    se2 = sigma_eps_sq.clamp(min=1e-12)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)                  # (B, m, d, q)
    beta_gls_fallback = torch.where(mx_rank[:, None] > 0, beta_wg, beta_ols)
    gls = _normalGlsAndBlups(
        Xm,
        ym,
        Zm,
        mask_n,
        ZtZ_safe,
        Zty,
        ZtX,
        XtX,
        Xty,
        XtZ,
        Psi,
        se2,
        eye_q,
        eye_q_bm,
        mask4,
        beta_gls_fallback,
    )
    weak_multi_q = (active_count > 1.0) & ~enough_full_mom
    beta_mask = torch.where(
        weak_multi_q[:, None],
        gls.beta_mask,
        gls.beta_mask.any(dim=-1, keepdim=True),
    )
    beta_rank = mx_rank.clamp(min=1.0, max=float(d))

    Psi, se2, gls = _emRefineNormal(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        ZtZ_safe,
        Zty,
        ZtX,
        XtX,
        Xty,
        XtZ,
        eye_q,
        eye_q_bm,
        mask4,
        mom4,
        G_mom,
        enough_full_mom,
        active_qq,
        beta_wg,
        beta_rank,
        beta_mask,
        Psi,
        se2,
        psi_diag_floor,
        psi_eig_cap,
        gls,
        n_em,
        uncorr,
    )

    beta_gls = gls.beta
    active_d_count = (XtX.diagonal(dim1=-2, dim2=-1).abs() > 1e-8).sum(dim=-1)
    blup_beta_alpha = torch.where(
        active_d_count <= 4,
        torch.ones_like(se2),
        torch.where(
            active_d_count <= 8, se2.new_full(se2.shape, 0.65), se2.new_full(se2.shape, 0.75)
        ),
    )
    beta_for_blup = (
        (1.0 - blup_beta_alpha[:, None]) * gls.beta + blup_beta_alpha[:, None] * beta_ols
    ).nan_to_num(
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_for_blup)) * mask_n
    Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)
    blups = torch.einsum('bmqp,bmp->bmq', gls.W_g, Ztr_gls)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
    W_ZtX = gls.W_ZtX
    A_gls_reg = gls.A_reg
    blup_var = gls.blup_var
    beta_identified = gls.beta_mask.any(dim=-1)

    # Inflate blup_var to account for uncertainty in the Psi estimate.
    # Var[Psi] ∝ Psi²/(G-d), delta-method gives 1 + 2/(G-d).
    # Clamp denominator at 2 (max 100% inflation) instead of 4; Fix 1 already handles large-n_g
    # over-confidence via the Ψ/G_mom additive floor, so the multiplicative cap can be looser.
    df_sigma = (G - d).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma.clamp(min=2.0))[:, None, None]

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
    # Ψ-uncertainty additive floor: when Ψ̂ is noisy (small G_mom), the dominant
    # blup prediction error source is Ψ̂/G_mom regardless of n_g. This term stays
    # finite for large groups where σ²W_g→0, preventing catastrophic overconfidence.
    blup_var_psi_floor = psi_diag[:, None, :] / G_mom.clamp(min=1.0)[:, None, None]
    blup_var = blup_var + blup_var_psi_floor

    corr_alpha = G / (G + 5.0)
    Psi = _shrinkOffDiagonal(Psi, corr_alpha)

    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)                 # (B, q)
    floor_tol = torch.maximum(psi_diag_floor.abs() * 1e-3, psi_diag_floor.new_full((), 1e-10))
    floor_hit = psi_diag <= psi_diag_floor + floor_tol
    floor_ratio = psi_diag / psi_diag_floor.clamp(min=1e-12)
    floor_limited = floor_hit & (active_count[:, None] > 2.0)
    q2_floor_limited = floor_hit & (active_count[:, None] == 2.0) & (floor_ratio <= 0.68)
    # Floor-pinned multi-component rows are systematically high, but lowering the runtime
    # floor destabilizes GLS. Calibrate only the reported marginal scale after BLUPs.
    psi_diag_out = torch.where(q2_floor_limited, 0.7225 * psi_diag, psi_diag)
    psi_diag_out = torch.where(floor_limited, 0.3025 * psi_diag, psi_diag_out)
    Psi_out = Psi + torch.diag_embed(psi_diag_out - psi_diag)
    sigma_rfx = psi_diag_out.sqrt()                                          # (B, q)
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
        'Psi': Psi_out,  # (B, q, q)
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
