"""MAP refinements for analytical GLMM summaries."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from metabeta.analytical.constants import _BERNOULLI_PSI_EIG_CAP
from metabeta.analytical.linalg import (
    _adaptiveRidge,
    _adaptiveRidgeBm,
    _psdClampEigenvalues,
    _psdProject,
    _pseudoInverse,
    _safeSolve,
)
from metabeta.analytical.normal import _normalGlsAndBlups
from metabeta.utils.families import logProbFfx, logProbSigma

__all__ = [
    'refineBernoulliLaplaceEb',
    'refineBernoulliMapBeta',
    'refineBernoulliNagqSrfx',
    'refineBernoulliNestedBeta',
    'refineNormalLaplaceEb',
    'refineNormalMapSrfx',
]


def _fixedCorrFromStats(
    stats: dict[str, torch.Tensor],
    eta_rfx: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    q: int,
) -> torch.Tensor:
    device = stats['Psi'].device
    dtype = stats['Psi'].dtype
    B = stats['Psi'].shape[0]
    eye = torch.eye(q, device=device, dtype=dtype).expand(B, q, q)
    Psi = stats['Psi'][..., :q, :q]
    std = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    corr = Psi / (std[:, :, None] * std[:, None, :])
    corr = corr.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.95, 0.95)
    corr = 0.5 * (corr + corr.mT)
    corr = corr - torch.diag_embed(corr.diagonal(dim1=-2, dim2=-1)) + eye

    if eta_rfx is not None:
        corr = torch.where(eta_rfx.to(device=device)[:, None, None] > 0, corr, eye)
    if mask_q is not None:
        mask = mask_q[..., :q].to(device=device).bool()
        corr = torch.where(mask[:, :, None] & mask[:, None, :], corr, eye)

    vals, vecs = torch.linalg.eigh(corr)
    corr = vecs @ torch.diag_embed(vals.clamp(min=1e-4)) @ vecs.mT
    corr_std = corr.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    corr = corr / (corr_std[:, :, None] * corr_std[:, None, :])
    return 0.5 * (corr + corr.mT)


def _psiFromSigmaCorr(sigma_rfx: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    return corr[:, None, :, :] * sigma_rfx[:, :, :, None] * sigma_rfx[:, :, None, :]


def _logMarginalLikelihoodNormal(
    beta: torch.Tensor,
    sigma_rfx: torch.Tensor,
    sigma_eps: torch.Tensor,
    corr: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
) -> torch.Tensor:
    y = ym.unsqueeze(-1)
    mask_n4 = mask_n.unsqueeze(-1)
    B, S, q = sigma_rfx.shape
    Psi = _psiFromSigmaCorr(sigma_rfx.clamp(min=1e-6), corr)
    eye_q = torch.eye(q, device=Psi.device, dtype=Psi.dtype)
    jitter = 1e-6 * eye_q

    chol_Psi, info_Psi = torch.linalg.cholesky_ex(Psi + jitter)
    psi_ok = info_Psi == 0
    Psi_inv = torch.cholesky_solve(eye_q.expand(B, S, q, q), chol_Psi)
    log_det_Psi = 2.0 * chol_Psi.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    Z_masked = Zm * mask_n4
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_masked, Z_masked)
    mu = torch.einsum('bmnd,bsd->bmns', Xm, beta)
    resid = (y - mu) * mask_n4
    rtr = resid.square().sum(dim=2)
    Ztr = torch.einsum('bmnq,bmns->bmsq', Z_masked, resid)
    n_i = mask_n.sum(dim=-1).float()

    s2e = sigma_eps.clamp(min=1e-6).square()
    M = Psi_inv[:, None] + ZtZ[:, :, None] / s2e[:, None, :, None, None]
    chol_M, info_M = torch.linalg.cholesky_ex(M + jitter)
    M_ok = info_M == 0
    log_det_M = 2.0 * chol_M.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    M_inv_Ztr = torch.cholesky_solve(Ztr.unsqueeze(-1), chol_M).squeeze(-1)
    hMh = (Ztr * M_inv_Ztr).sum(-1)

    log_det_V = log_det_M + log_det_Psi[:, None, :] + n_i[:, :, None] * s2e.log()[:, None, :]
    quad = (rtr - hMh / s2e[:, None, :]) / s2e[:, None, :]
    ll_i = -0.5 * (n_i[:, :, None] * math.log(2.0 * math.pi) + log_det_V + quad)
    active_m = mask_m.bool()[:, :, None]
    ll_i = torch.where((M_ok & psi_ok[:, None, :]) | ~active_m, ll_i, ll_i.new_tensor(-torch.inf))
    return torch.where(active_m, ll_i, ll_i.new_zeros(())).sum(dim=1)


def _logMarginalTarget(
    beta: torch.Tensor,
    log_sigma_rfx: torch.Tensor,
    log_sigma_eps: torch.Tensor,
    corr: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor,
    tau_ffx: torch.Tensor,
    family_ffx: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    tau_eps: torch.Tensor,
    family_sigma_eps: torch.Tensor,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
) -> torch.Tensor:
    sigma_rfx = log_sigma_rfx.exp()
    sigma_eps = log_sigma_eps.exp()
    ll = _logMarginalLikelihoodNormal(beta, sigma_rfx, sigma_eps, corr, Xm, ym, Zm, mask_n, mask_m)

    d = beta.shape[-1]
    q = sigma_rfx.shape[-1]
    if mask_d is None:
        mask_d = torch.ones(beta.shape[0], d, dtype=torch.bool, device=beta.device)
    if mask_q is None:
        mask_q = torch.ones(beta.shape[0], q, dtype=torch.bool, device=beta.device)
    mask_d_lp = mask_d[..., :d].unsqueeze(-2).to(beta.dtype)
    mask_q_lp = mask_q[..., :q].unsqueeze(-2).to(beta.dtype)

    lp = logProbFfx(
        beta,
        nu_ffx[..., :d].unsqueeze(-2),
        tau_ffx[..., :d].unsqueeze(-2) + 1e-12,
        family_ffx,
        mask_d_lp,
    )
    lp = lp + logProbSigma(
        sigma_rfx,
        tau_rfx[..., :q].unsqueeze(-2) + 1e-12,
        family_sigma_rfx,
        mask_q_lp,
    )
    lp = lp + logProbSigma(
        sigma_eps,
        tau_eps.unsqueeze(-1) + 1e-12,
        family_sigma_eps,
    )
    return ll + lp


def _replacePsiDiag(
    Psi: torch.Tensor,
    sigma_rfx: torch.Tensor,
    mask_q: torch.Tensor | None,
) -> torch.Tensor:
    q = sigma_rfx.shape[-1]
    Psi_q = Psi[..., :q, :q]
    old_std = Psi_q.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    corr = (Psi_q / (old_std[:, :, None] * old_std[:, None, :])).nan_to_num(
        nan=0.0, posinf=0.0, neginf=0.0
    )
    corr = corr.clamp(-0.95, 0.95)
    eye = torch.eye(q, dtype=Psi.dtype, device=Psi.device)
    corr = corr - torch.diag_embed(corr.diagonal(dim1=-2, dim2=-1)) + eye
    refined = corr * sigma_rfx[:, :, None] * sigma_rfx[:, None, :]
    if mask_q is not None:
        active = mask_q[..., :q].bool()
        refined = torch.where(active[:, :, None] & active[:, None, :], refined, Psi_q)
    return Psi + F.pad(refined - Psi_q, (0, Psi.shape[-1] - q, 0, Psi.shape[-2] - q))


def _normalSigmaGridBetaAverage(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    sigma_rfx_base: torch.Tensor,
    sigma_eps: torch.Tensor,
    nu_ffx: torch.Tensor,
    tau_ffx: torch.Tensor,
    family_ffx: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    tau_eps: torch.Tensor,
    family_sigma_eps: torch.Tensor,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    scales: tuple[float, ...] | list[float],
) -> torch.Tensor:
    """Approximate hyperparameter-averaged β over a small diagonal-σ grid."""
    q = Zm.shape[-1]
    d = Xm.shape[-1]
    if mask_q is not None:
        Zm = Zm * mask_q[..., :q].to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :]

    clean_scales = [float(scale) for scale in scales if float(scale) > 0.0]
    if not clean_scales:
        clean_scales = [1.0]

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype
    S = len(clean_scales)
    scale_tensor = torch.tensor(clean_scales, device=device, dtype=dtype)
    sigma_grid = (sigma_rfx_base[:, None, :] * scale_tensor[None, :, None]).clamp(
        min=1e-4, max=20.0
    )
    se2 = sigma_eps.clamp(min=1e-6).square()

    mask4 = mask_n[:, :, :, None]
    Xm_masked = Xm * mask4
    Zm_masked = Zm * mask4
    active = mask_m.bool()
    eye_q = torch.eye(q, device=device, dtype=dtype)

    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm_masked, Zm_masked)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)
    Zty = torch.einsum('bmnq,bmn->bmq', Zm_masked, ym)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm_masked, Xm_masked)
    XtZ = ZtX.mT
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm_masked, Xm_masked)
    Xty = torch.einsum('bmnd,bmn->bd', Xm_masked, ym)

    sigma2_grid = sigma_grid.square().clamp(min=1e-8)
    inner = ZtZ_safe[:, None] + torch.diag_embed(
        se2[:, None, None, None] / sigma2_grid[:, :, None, :]
    )
    eye_q_bsm = eye_q.expand(B, S, m, q, q)
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bsm)
    W_ZtX = torch.einsum('bsmqp,bmpd->bsmqd', W_g, ZtX)
    correction_XX = torch.einsum('bmdq,bsmqk->bsdk', XtZ, W_ZtX)
    W_Zty = torch.einsum('bsmqp,bmp->bsmq', W_g, Zty)
    correction_Xy = torch.einsum('bmdq,bsmq->bsd', XtZ, W_Zty)

    A = (XtX[:, None] - correction_XX) / se2[:, None, None, None]
    b = (Xty[:, None] - correction_Xy) / se2[:, None, None]
    prior_prec = 1.0 / tau_ffx[..., :d].clamp(min=1e-4).square()
    A = A + torch.diag_embed(prior_prec[:, None, :])
    b = b + prior_prec[:, None, :] * nu_ffx[..., :d][:, None, :]
    beta_grid = _safeSolve(A + _adaptiveRidge(A), b).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta_grid = beta_grid.clamp(-20.0, 20.0)

    corr = torch.eye(q, device=device, dtype=dtype).expand(B, q, q)
    target = _logMarginalTarget(
        beta_grid,
        sigma_grid.log(),
        sigma_eps.clamp(min=1e-4, max=20.0).log()[:, None].expand(B, S),
        corr,
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        nu_ffx,
        tau_ffx,
        family_ffx,
        tau_rfx,
        family_sigma_rfx,
        tau_eps,
        family_sigma_eps,
        mask_d,
        mask_q,
    )
    weights = torch.softmax(target.nan_to_num(nan=-1e30, posinf=1e30, neginf=-1e30), dim=1)
    beta_avg = (weights[:, :, None] * beta_grid).sum(dim=1)
    if mask_d is not None:
        active_d = mask_d[..., :d].bool()
        beta_avg = torch.where(active_d, beta_avg, beta_grid[:, 0, :])
    return beta_avg


def _normalSigmaRfxGridRefine(
    beta: torch.Tensor,
    sigma_rfx_base: torch.Tensor,
    log_sigma_eps: torch.Tensor,
    current_target: torch.Tensor,
    corr: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor,
    tau_ffx: torch.Tensor,
    family_ffx: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    tau_eps: torch.Tensor,
    family_sigma_eps: torch.Tensor,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    scales: tuple[float, ...] | list[float],
    accept_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate grid refinement of diagonal σ_rfx under the marginal target."""
    q = sigma_rfx_base.shape[-1]
    clean_scales = sorted({float(scale) for scale in scales if float(scale) > 0.0})
    if not clean_scales:
        clean_scales = [1.0]
    if all(abs(scale - 1.0) <= 1e-8 for scale in clean_scales):
        return sigma_rfx_base, current_target, torch.zeros_like(current_target, dtype=torch.bool)

    B = sigma_rfx_base.shape[0]
    device, dtype = sigma_rfx_base.device, sigma_rfx_base.dtype
    scale_tensor = torch.tensor(clean_scales, device=device, dtype=dtype)
    S = scale_tensor.numel()
    active_q = (
        mask_q[..., :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )

    sigma_rfx = sigma_rfx_base.clamp(min=1e-4, max=20.0).clone()
    target = current_target.clone()
    changed = torch.zeros(B, device=device, dtype=torch.bool)

    for j in range(q):
        active_j = active_q[:, j]
        if not bool(active_j.any()):
            continue
        candidates = sigma_rfx[:, None, :].expand(B, S, q).clone()
        candidates[:, :, j] = (sigma_rfx[:, j, None] * scale_tensor[None]).clamp(min=1e-4, max=20.0)
        candidate_target = _logMarginalTarget(
            beta[:, None, :].expand(B, S, beta.shape[-1]),
            candidates.log(),
            log_sigma_eps[:, None].expand(B, S),
            corr,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            nu_ffx,
            tau_ffx,
            family_ffx,
            tau_rfx,
            family_sigma_rfx,
            tau_eps,
            family_sigma_eps,
            mask_d,
            mask_q,
        )
        candidate_target = candidate_target.nan_to_num(
            nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
        )
        best_target, best_idx = candidate_target.max(dim=1)
        improve = active_j & torch.isfinite(best_target) & (best_target > target + accept_tol)
        sigma_j = candidates[torch.arange(B, device=device), best_idx, j]
        sigma_rfx[:, j] = torch.where(improve, sigma_j, sigma_rfx[:, j])
        target = torch.where(improve, best_target, target)
        changed |= improve

    sigma_rfx = torch.where(active_q, sigma_rfx, sigma_rfx_base)
    return sigma_rfx, target, changed


def _recomputeNormalFinalDiagMap(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    sigma_rfx: torch.Tensor,
    mask_q: torch.Tensor | None,
    beta_alpha_low: float = 0.65,
    beta_alpha_high: float = 0.75,
    beta_override: torch.Tensor | None = None,
    beta_for_blup_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Recompute final Gaussian GLS/BLUP using diagonal MAP Psi."""
    q = Zm.shape[-1]
    if mask_q is not None:
        Zm = Zm * mask_q[..., :q].to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :]

    B, m, _, d = Xm.shape
    device = Xm.device
    dtype = Xm.dtype
    active = mask_m.bool()
    mask4 = mask_m[:, :, None, None]
    eye_q = torch.eye(q, device=device, dtype=dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)

    Psi = torch.diag_embed(sigma_rfx.square())
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)
    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)
    se2 = stats['sigma_eps_est'].squeeze(-1).clamp(min=1e-6).square()
    beta_fallback = stats.get('beta_wg', stats['beta_est'])

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
        beta_fallback,
    )

    if beta_override is None:
        beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
        active_d_count = (XtX.diagonal(dim1=-2, dim2=-1).abs() > 1e-8).sum(dim=-1)
        alpha = torch.where(
            active_d_count <= 8,
            se2.new_full(se2.shape, beta_alpha_low),
            se2.new_full(se2.shape, beta_alpha_high),
        )
        beta_final = gls.beta
        beta_for_blup = ((1.0 - alpha[:, None]) * gls.beta + alpha[:, None] * beta_ols).nan_to_num(
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        beta_final = beta_override.detach().to(device=device, dtype=dtype)
        if beta_for_blup_override is None:
            beta_for_blup = beta_final
        else:
            beta_for_blup = beta_for_blup_override.detach().to(device=device, dtype=dtype)
        beta_for_blup = beta_for_blup.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_for_blup)) * mask_n
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    blups = torch.einsum('bmqp,bmp->bmq', gls.W_g, Ztr)

    # Recompute blup_var from MAP GLS quantities (σ²·MAP_W_g diagonal).
    # The raw blup_var uses raw Psi, which understates MAP shrinkage and
    # produces inflated uncertainty via the Ψ/G_mom additive floor.
    blup_var = gls.blup_var  # (B, m, q), already clamped/nan-cleaned

    # Delta-method inflation for Psi estimation uncertainty: 1 + 2/(G-d).
    G = mask_m.float().sum(dim=-1)  # (B,)
    df_sigma = (G - float(d)).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma.clamp(min=2.0))[:, None, None]

    # Kackar-Harville correction: beta uncertainty propagation into BLUPs.
    beta_identified = gls.beta_mask.any(dim=-1)  # (B,)
    gls_determined = G >= float(d)
    vals_kh, vecs_kh = torch.linalg.eigh(gls.A_reg)
    max_kh = vals_kh.amax(dim=-1, keepdim=True).clamp(min=1.0)
    inv_vals_kh = torch.where(
        vals_kh > 1e-3 * max_kh,
        1.0 / vals_kh.clamp(min=1e-30),
        torch.zeros_like(vals_kh),
    )
    beta_var_kh = (vecs_kh**2 * inv_vals_kh[:, None, :]).sum(dim=-1).clamp(min=1e-8)
    kh_corr = (gls.W_ZtX**2 * beta_var_kh[:, None, None, :]).sum(dim=-1)
    kh_corr = kh_corr * (beta_identified & gls_determined)[:, None, None]
    blup_var = blup_var + kh_corr

    # n_g-dependent floor: prevents near-zero variance for small groups.
    # The Ψ/G_mom additive floor from the raw path is omitted: under MAP,
    # Psi is better estimated so the large floor (calibrated for raw scale)
    # would overstate uncertainty and dominate for medium-to-large groups.
    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    blup_var_floor = psi_diag[:, None, :] / (2.0 * ns.clamp(min=1.0)[:, :, None])
    blup_var = blup_var.clamp(min=blup_var_floor)

    out = dict(stats)
    out['beta_est'] = beta_final
    out['sigma_rfx_est'] = sigma_rfx
    out['blup_est'] = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
    out['blup_var'] = blup_var
    out['Psi'] = Psi
    return out


def _guardNormalAliasedBlups(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    sigma_rfx: torch.Tensor,
    tau_rfx: torch.Tensor,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    beta_base: torch.Tensor,
    beta_output: torch.Tensor,
    sigma_prior_cap: float = 4.0,
    blup_prior_ratio: float = 4.0,
    min_d: int = 13,
) -> dict[str, torch.Tensor]:
    """Fallback for rare high-d aliased rows with implausibly large BLUPs."""
    q = Zm.shape[-1]
    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    active_m = mask_m.bool()
    active_q = (
        mask_q[..., :q].bool()
        if mask_q is not None
        else torch.ones((B, q), device=device, dtype=torch.bool)
    )
    active = active_m[:, :, None] & active_q[:, None, :]
    denom = active.to(dtype).sum(dim=(1, 2)).clamp(min=1.0)
    blup = stats['blup_est'][..., :q]
    blup_rms = ((blup.square() * active.to(dtype)).sum(dim=(1, 2)) / denom).sqrt()

    tau = tau_rfx[..., :q].to(device=device, dtype=dtype).clamp(min=1e-4)
    tau_rms = (
        (tau.square() * active_q.to(dtype)).sum(dim=-1) / active_q.sum(dim=-1).clamp(min=1)
    ).sqrt()
    sigma_active = sigma_rfx[..., :q].to(device=device, dtype=dtype).clamp(min=1e-4)
    sigma_inflated = (sigma_active > float(sigma_prior_cap) * tau) & active_q

    if mask_d is not None:
        d_count = mask_d.bool().sum(dim=-1)
    else:
        d_count = torch.full((B,), Xm.shape[-1], device=device, dtype=torch.long)
    guard = (
        (d_count >= int(min_d))
        & sigma_inflated.any(dim=-1)
        & (blup_rms > float(blup_prior_ratio) * tau_rms.clamp(min=1e-4))
    )
    if not guard.any():
        out = dict(stats)
        out['normal_laplace_eb_blup_guard'] = torch.zeros((B,), device=device, dtype=dtype)
        return out

    sigma_guard = torch.minimum(sigma_active, float(sigma_prior_cap) * tau)
    sigma_guard = torch.where(active_q, sigma_guard, sigma_active)
    sigma_next = torch.where(guard[:, None], sigma_guard, sigma_active)

    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
    beta_ols = beta_ols.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta_for_blup = torch.where(guard[:, None], beta_ols, beta_base)

    out = _recomputeNormalFinalDiagMap(
        stats,
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        sigma_next,
        mask_q,
        beta_override=beta_output,
        beta_for_blup_override=beta_for_blup,
    )
    out['normal_laplace_eb_blup_guard'] = guard.to(dtype)
    return out


def refineNormalMapSrfx(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    nu_ffx: torch.Tensor,
    tau_ffx: torch.Tensor,
    family_ffx: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    tau_eps: torch.Tensor,
    family_sigma_eps: torch.Tensor,
    eta_rfx: torch.Tensor | None = None,
    mask_d: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    n_steps: int = 20,
    lr: float = 0.03,
    recompute_blup: bool = True,
    optimize: str = 'all',
    beta_alpha_low: float = 0.65,
    beta_alpha_high: float = 0.75,
    beta_prior_cap: float | None = 4.0,
    beta_sigma_grid: bool = False,
    beta_sigma_grid_scales: tuple[float, ...] | list[float] = (0.75, 1.0, 1.3333333),
    beta_sigma_grid_min_d: int = 5,
) -> dict[str, torch.Tensor]:
    """Return stats with sigma(RFX) refined by marginal MAP.

    By default, final Gaussian GLS/BLUP is recomputed with a diagonal MAP Psi.
    This keeps the useful MAP sigma(RFX) scale while excluding noisy estimated
    correlations from the final BLUP shrinkage covariance.

    optimize controls which parameters receive gradients during MAP optimization.
    Supported values: 'rfx' (sigma_rfx only), 'rfx_beta', 'rfx_eps', 'all'.
    """
    q = Zm.shape[-1]
    if q == 0 or n_steps <= 0:
        return stats
    corr = _fixedCorrFromStats(stats, eta_rfx, mask_q, q)

    opt_beta = optimize in ('rfx_beta', 'all')
    opt_eps = optimize in ('rfx_eps', 'all')

    beta = stats['beta_est'].detach().clone().requires_grad_(opt_beta)
    log_sigma_rfx = (
        stats['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    log_sigma_eps = (
        stats['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(opt_eps)

    opt_params = [log_sigma_rfx]
    if opt_beta:
        opt_params.append(beta)
    if opt_eps:
        opt_params.append(log_sigma_eps)
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            target = _logMarginalTarget(
                beta.unsqueeze(1),
                log_sigma_rfx.unsqueeze(1),
                log_sigma_eps.unsqueeze(1),
                corr,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx,
                tau_ffx,
                family_ffx,
                tau_rfx,
                family_sigma_rfx,
                tau_eps,
                family_sigma_eps,
                mask_d,
                mask_q,
            ).squeeze(1)
            loss = -target.sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([beta, log_sigma_rfx, log_sigma_eps], max_norm=10.0)
            optimizer.step()
            with torch.no_grad():
                beta.clamp_(-20.0, 20.0)
                log_sigma_rfx.clamp_(math.log(1e-4), math.log(20.0))
                log_sigma_eps.clamp_(math.log(1e-4), math.log(20.0))

    sigma_rfx = log_sigma_rfx.detach().exp()
    if mask_q is not None:
        sigma_rfx = torch.where(mask_q[..., :q].bool(), sigma_rfx, stats['sigma_rfx_est'][..., :q])
    out = dict(stats)
    # MAP beta overfits the low-d normal sets; keep the older GLS/OLS blend there.
    beta_for_blup_override = beta.detach() if opt_beta and beta.shape[-1] > 4 else None
    beta_override = beta_for_blup_override
    if beta_override is not None:
        beta_capped = torch.zeros(
            beta_override.shape[0], device=beta_override.device, dtype=beta_override.dtype
        )
        beta_stabilized = torch.zeros_like(beta_capped)
        if beta_prior_cap is not None and beta_prior_cap > 0:
            d = beta_override.shape[-1]
            cap_scale = float(beta_prior_cap)
            nu_d = nu_ffx[..., :d]
            tau_d = tau_ffx[..., :d].clamp(min=1e-4)
            cap_lo = nu_d - cap_scale * tau_d
            cap_hi = nu_d + cap_scale * tau_d
            cap_components = (beta_override < cap_lo) | (beta_override > cap_hi)
            beta_capped = cap_components.any(dim=-1).to(beta_override.dtype)
            beta_override = beta_override.clamp(min=cap_lo, max=cap_hi)
            if mask_d is not None:
                active_d = mask_d[..., :d].bool()
                beta_override = torch.where(active_d, beta_override, beta.detach())
                cap_components = cap_components & active_d
                d_count = active_d.sum(dim=-1)
            else:
                d_count = torch.full(
                    (beta_override.shape[0],), d, device=beta_override.device, dtype=torch.long
                )
            if beta_sigma_grid:
                stabilize = cap_components & (d_count >= int(beta_sigma_grid_min_d))[:, None]
                beta_grid = _normalSigmaGridBetaAverage(
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    sigma_rfx,
                    stats['sigma_eps_est'].squeeze(-1).detach(),
                    nu_ffx,
                    tau_ffx,
                    family_ffx,
                    tau_rfx,
                    family_sigma_rfx,
                    tau_eps,
                    family_sigma_eps,
                    mask_d,
                    mask_q,
                    beta_sigma_grid_scales,
                )
                beta_grid = beta_grid.clamp(min=cap_lo, max=cap_hi)
                beta_override = torch.where(stabilize, beta_grid, beta_override)
                beta_stabilized = stabilize.any(dim=-1).to(beta_override.dtype)
        out['beta_est'] = beta_override
        out['normal_map_beta_for_blup'] = beta_for_blup_override
        out['normal_map_beta_prior_capped'] = beta_capped
        out['normal_map_beta_stabilized'] = beta_stabilized
    out['sigma_rfx_est'] = sigma_rfx
    if 'Psi' in stats:
        if recompute_blup:
            return _recomputeNormalFinalDiagMap(
                out,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                ns,
                sigma_rfx,
                mask_q,
                beta_alpha_low=beta_alpha_low,
                beta_alpha_high=beta_alpha_high,
                beta_override=beta_override,
                beta_for_blup_override=beta_for_blup_override,
            )
        out['Psi'] = _replacePsiDiag(stats['Psi'], sigma_rfx, mask_q)
    return out


def refineNormalLaplaceEb(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    nu_ffx: torch.Tensor,
    tau_ffx: torch.Tensor,
    family_ffx: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    tau_eps: torch.Tensor,
    family_sigma_eps: torch.Tensor,
    mask_d: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    n_steps: int = 3,
    lr: float = 0.08,
    mode: str = 'moment',
    moment_blend: float = 1.0,
    prior_weight: float = 4.0,
    optimize_eps: bool = False,
    recompute_blup: bool = True,
    beta_alpha_low: float = 0.65,
    beta_alpha_high: float = 0.75,
    accept_tol: float = 1e-6,
    sigma_grid_refine: bool = False,
    sigma_grid_scales: tuple[float, ...] | list[float] = (0.75, 1.0, 1.3333333),
) -> dict[str, torch.Tensor]:
    """Prototype diagonal Laplace-EB calibration for Gaussian GLMMs.

    Gaussian random-effect integration is exact, so this only adjusts diagonal variance
    scales under the exact marginal likelihood and priors. β stays fixed; final β/BLUPs
    are recomputed once through the existing diagonal Gaussian pass.
    """
    q = Zm.shape[-1]
    if q == 0 or n_steps <= 0:
        return stats

    if mask_q is not None:
        Zm = Zm * mask_q[..., :q].to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :]

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    beta = stats.get('normal_map_beta_for_blup', stats['beta_est']).detach()
    beta_output = stats['beta_est'].detach()
    sigma_rfx_base = stats['sigma_rfx_est'][..., :q].detach().clamp(min=1e-4, max=20.0)
    log_sigma_eps_base = stats['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log()
    corr = torch.eye(q, device=device, dtype=dtype).expand(B, q, q)

    def target_for(sigma_rfx: torch.Tensor, log_sigma_eps_value: torch.Tensor) -> torch.Tensor:
        return _logMarginalTarget(
            beta.unsqueeze(1),
            sigma_rfx.clamp(min=1e-4, max=20.0).log().unsqueeze(1),
            log_sigma_eps_value.unsqueeze(1),
            corr,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            nu_ffx,
            tau_ffx,
            family_ffx,
            tau_rfx,
            family_sigma_rfx,
            tau_eps,
            family_sigma_eps,
            mask_d,
            mask_q,
        ).squeeze(1)

    if mode == 'moment':
        with torch.no_grad():
            base_target = target_for(sigma_rfx_base, log_sigma_eps_base)
            sigma2_base = sigma_rfx_base.square().clamp(min=1e-8)
            se2 = stats['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-6).square()
            active = mask_m.bool()
            eye_q = torch.eye(q, device=device, dtype=dtype)
            eye_q_bm = eye_q.expand(B, Xm.shape[1], q, q)
            ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
            inner = ZtZ + torch.diag_embed(se2[:, None, None] / sigma2_base[:, None, :])
            inner = torch.where(active[:, :, None, None], inner, eye_q)
            W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm)

            XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
            Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
            beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
            active_d_count = (XtX.diagonal(dim1=-2, dim2=-1).abs() > 1e-8).sum(dim=-1)
            alpha = torch.where(
                active_d_count <= 8,
                se2.new_full(se2.shape, beta_alpha_low),
                se2.new_full(se2.shape, beta_alpha_high),
            )
            beta_for_blup = (1.0 - alpha[:, None]) * beta + alpha[:, None] * beta_ols
            resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_for_blup)) * mask_n
            Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
            blup = torch.einsum('bmqp,bmp->bmq', W_g, Ztr).clamp(-20.0, 20.0)
            post_var = se2[:, None, None] * W_g.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
            active_f = active.to(dtype)
            G = active_f.sum(dim=-1).clamp(min=1.0)
            moment2 = ((blup.square() + post_var) * active_f[:, :, None]).sum(dim=1)
            tau2 = tau_rfx[..., :q].to(device=device, dtype=dtype).clamp(min=1e-4).square()
            moment2 = (moment2 + prior_weight * tau2) / (G[:, None] + prior_weight)
            sigma_moment = moment2.clamp(min=1e-8, max=400.0).sqrt()
            blend = max(0.0, min(1.0, float(moment_blend)))
            sigma_candidate = (
                (1.0 - blend) * sigma_rfx_base.log() + blend * sigma_moment.log()
            ).exp()
            final_target = target_for(sigma_candidate, log_sigma_eps_base)
            accept = torch.isfinite(final_target) & (final_target >= base_target - accept_tol)
            sigma_rfx = torch.where(accept[:, None], sigma_candidate, sigma_rfx_base)
            if mask_q is not None:
                active_q = mask_q[..., :q].bool()
                sigma_rfx = torch.where(active_q, sigma_rfx, sigma_rfx_base)
            grid_changed = torch.zeros(B, device=device, dtype=torch.bool)
            if sigma_grid_refine:
                sigma_rfx, final_target, grid_changed = _normalSigmaRfxGridRefine(
                    beta,
                    sigma_rfx,
                    log_sigma_eps_base,
                    torch.where(accept, final_target, base_target),
                    corr,
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    nu_ffx,
                    tau_ffx,
                    family_ffx,
                    tau_rfx,
                    family_sigma_rfx,
                    tau_eps,
                    family_sigma_eps,
                    mask_d,
                    mask_q,
                    sigma_grid_scales,
                    accept_tol=accept_tol,
                )

        out = dict(stats)
        out['sigma_rfx_est'] = sigma_rfx
        out['normal_laplace_eb_accept'] = accept.to(dtype)
        out['normal_laplace_eb_sigma_grid_accept'] = grid_changed.to(dtype)
        out['normal_laplace_eb_steps'] = torch.zeros((B,), device=device, dtype=dtype)
        out['normal_laplace_eb_target'] = final_target.detach()
        out['normal_laplace_eb_base_target'] = base_target.detach()
        if 'Psi' in stats:
            if recompute_blup:
                out = _recomputeNormalFinalDiagMap(
                    out,
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    ns,
                    sigma_rfx,
                    mask_q,
                    beta_alpha_low=beta_alpha_low,
                    beta_alpha_high=beta_alpha_high,
                    beta_override=beta_output if beta.shape[-1] > 4 else None,
                    beta_for_blup_override=beta if beta.shape[-1] > 4 else None,
                )
                return _guardNormalAliasedBlups(
                    out,
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    ns,
                    sigma_rfx,
                    tau_rfx,
                    mask_d,
                    mask_q,
                    beta,
                    beta_output,
                )
            out['Psi'] = torch.diag_embed(sigma_rfx.square())
        return out

    if mode != 'gradient':
        raise ValueError("normal_laplace_eb mode must be 'moment' or 'gradient'")

    log_sigma_rfx = (sigma_rfx_base.log().clone()).requires_grad_(True)
    log_sigma_eps = log_sigma_eps_base.clone().requires_grad_(optimize_eps)

    with torch.no_grad():
        base_target = target_for(sigma_rfx_base, log_sigma_eps.detach())

    opt_params = [log_sigma_rfx]
    if optimize_eps:
        opt_params.append(log_sigma_eps)
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    n_steps_run = 0
    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            target = _logMarginalTarget(
                beta.unsqueeze(1),
                log_sigma_rfx.unsqueeze(1),
                log_sigma_eps.unsqueeze(1),
                corr,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx,
                tau_ffx,
                family_ffx,
                tau_rfx,
                family_sigma_rfx,
                tau_eps,
                family_sigma_eps,
                mask_d,
                mask_q,
            ).squeeze(1)
            finite = torch.isfinite(target)
            if not finite.any():
                break
            loss = -target[finite].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=10.0)
            optimizer.step()
            n_steps_run += 1
            with torch.no_grad():
                log_sigma_rfx.clamp_(math.log(1e-4), math.log(20.0))
                log_sigma_eps.clamp_(math.log(1e-4), math.log(20.0))

    with torch.no_grad():
        final_target = _logMarginalTarget(
            beta.unsqueeze(1),
            log_sigma_rfx.detach().unsqueeze(1),
            log_sigma_eps.detach().unsqueeze(1),
            corr,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            nu_ffx,
            tau_ffx,
            family_ffx,
            tau_rfx,
            family_sigma_rfx,
            tau_eps,
            family_sigma_eps,
            mask_d,
            mask_q,
        ).squeeze(1)
        accept = torch.isfinite(final_target) & (final_target >= base_target - accept_tol)

    sigma_rfx_new = log_sigma_rfx.detach().exp()
    sigma_rfx = torch.where(accept[:, None], sigma_rfx_new, sigma_rfx_base)
    if mask_q is not None:
        active_q = mask_q[..., :q].bool()
        sigma_rfx = torch.where(active_q, sigma_rfx, sigma_rfx_base)
    grid_changed = torch.zeros(B, device=device, dtype=torch.bool)
    if sigma_grid_refine:
        log_sigma_eps_grid = torch.where(accept, log_sigma_eps.detach(), log_sigma_eps_base)
        sigma_rfx, final_target, grid_changed = _normalSigmaRfxGridRefine(
            beta,
            sigma_rfx,
            log_sigma_eps_grid,
            torch.where(accept, final_target, base_target),
            corr,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            nu_ffx,
            tau_ffx,
            family_ffx,
            tau_rfx,
            family_sigma_rfx,
            tau_eps,
            family_sigma_eps,
            mask_d,
            mask_q,
            sigma_grid_scales,
            accept_tol=accept_tol,
        )

    out = dict(stats)
    if optimize_eps:
        sigma_eps_new = log_sigma_eps.detach().exp()
        sigma_eps_base = stats['sigma_eps_est'].squeeze(-1)
        out['sigma_eps_est'] = torch.where(accept, sigma_eps_new, sigma_eps_base).unsqueeze(-1)
    out['sigma_rfx_est'] = sigma_rfx
    out['normal_laplace_eb_accept'] = accept.to(dtype)
    out['normal_laplace_eb_sigma_grid_accept'] = grid_changed.to(dtype)
    out['normal_laplace_eb_steps'] = torch.full(
        (B,), float(n_steps_run), device=device, dtype=dtype
    )
    out['normal_laplace_eb_target'] = final_target.detach()
    out['normal_laplace_eb_base_target'] = base_target.detach()

    if 'Psi' in stats:
        if recompute_blup:
            out = _recomputeNormalFinalDiagMap(
                out,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                ns,
                sigma_rfx,
                mask_q,
                beta_alpha_low=beta_alpha_low,
                beta_alpha_high=beta_alpha_high,
                beta_override=beta_output if beta.shape[-1] > 4 else None,
                beta_for_blup_override=beta if beta.shape[-1] > 4 else None,
            )
            return _guardNormalAliasedBlups(
                out,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                ns,
                sigma_rfx,
                tau_rfx,
                mask_d,
                mask_q,
                beta,
                beta_output,
            )
        out['Psi'] = torch.diag_embed(sigma_rfx.square())
    return out


def refineBernoulliMapBeta(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor | None = None,
    tau_ffx: torch.Tensor | None = None,
    family_ffx: torch.Tensor | None = None,
    n_steps: int = 8,
    n_newton: int = 3,
    n_outer: int = 2,
    damping: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Refine β via Newton on the true Bernoulli score at fixed b̂_g, then update BLUPs.

    PQL solves for β via a Schur-complement GLS on the linearized working response,
    which introduces approximation error that dominates FFX NRMSE.  This fixes b̂_g
    from PQL and runs Newton steps on the exact score ∑_g X_g'(y_g − σ(Xβ + Zb̂_g)),
    then recomputes b̂_g with the improved β (Ψ held fixed at the PQL Laplace M-step).

    n_outer controls how many times the β→b̂_g alternation is repeated.  Each outer
    iteration refines β at the current b̂_g (n_steps Newton steps) then recomputes b̂_g
    at the new β (n_newton Newton steps).  The Ψ M-step runs once after all outer
    iterations to update σ_rfx from the final b̂_g.

    Note: running to full convergence (large n_steps/n_outer) is harmful — the MAP
    (β, b̂) given the PQL Ψ is worse than early-stopped Newton because PQL Ψ is
    biased, and b̂ compensates for β at the MAP under the wrong Ψ.  The fixed budget
    here acts as beneficial implicit regularization via early stopping.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or n_steps <= 0 or n_outer <= 0:
        return stats

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype

    # Prior setup mirrors irlsBernoulli
    active_d_mask = None
    normal_prec = None
    is_student = None
    zeros_d = None
    if nu_ffx is not None and tau_ffx is not None:
        active_d_mask = tau_ffx > 0
        zeros_d = tau_ffx.new_zeros(tau_ffx.shape)
        normal_prec = torch.where(active_d_mask, 1.0 / tau_ffx.clamp(min=1e-4).square(), zeros_d)
        is_student = (family_ffx == 1).unsqueeze(-1) if family_ffx is not None else None

    has_rfx = q > 0 and n_newton > 0 and 'Psi_lap' in stats
    if has_rfx:
        Psi_lap = stats['Psi_lap']
        Psi_inv = _pseudoInverse(Psi_lap)
        eye_q = torch.eye(q, device=device, dtype=dtype)
        eye_q_bm = eye_q.expand(B, m, q, q)
        active = mask_m.bool()
        mask4 = mask_m[:, :, None, None]
        G = mask_m.sum(dim=1).clamp(min=1.0)

    blups = stats['blup_est'].detach().clone()  # (B, m, q) — updated each outer iter
    beta = stats['beta_est'].detach().clone()   # (B, d)

    for _outer in range(n_outer):
        # --- β Newton: n_steps steps at current b̂_g ---
        eta_rfx = torch.einsum('bmnq,bmq->bmn', Zm, blups)  # (B, m, n)
        for _ in range(n_steps):
            eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + eta_rfx
            mu = torch.sigmoid(eta)
            w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n                # (B, m, n)
            XtWX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)         # (B, d, d)
            score = torch.einsum('bmnd,bmn->bd', Xm, (ym - mu) * mask_n)  # (B, d)

            if normal_prec is not None:
                if is_student is not None:
                    student_prec = torch.where(
                        active_d_mask,
                        6.0
                        / (5.0 * tau_ffx.clamp(min=1e-8).square() + (beta - nu_ffx).square()).clamp(
                            min=1e-8
                        ),
                        zeros_d,
                    )
                    prior_prec = torch.where(is_student, student_prec, normal_prec)
                else:
                    prior_prec = normal_prec
                XtWX = XtWX + torch.diag_embed(prior_prec)
                score = score + prior_prec * (nu_ffx - beta)

            delta = _safeSolve(XtWX + _adaptiveRidge(XtWX), score)
            beta = (
                (beta + damping * delta)
                .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                .clamp(-50.0, 50.0)
            )

        if not has_rfx:
            continue

        # --- b̂_g Newton: n_newton steps at current β (Ψ fixed) ---
        for _ in range(n_newton):
            eta_b = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum(
                'bmnq,bmq->bmn', Zm, blups
            )
            mu_b = torch.sigmoid(eta_b)
            w_b = (mu_b * (1.0 - mu_b)).clamp(min=1e-6) * mask_n
            score_g = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_b) * mask_n)
            score_g = score_g - torch.einsum('bqr,bmr->bmq', Psi_inv, blups)
            ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_b, Zm)
            ZWZ_safe = torch.where(active[:, :, None, None], ZWZ, eye_q_bm)
            Hg = ZWZ_safe + Psi_inv[:, None]
            delta_b = _safeSolve(Hg + _adaptiveRidgeBm(Hg), score_g)
            blups = (blups + damping * delta_b) * mask_m[:, :, None]
            blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    out = dict(stats)
    out['beta_est'] = beta

    if not has_rfx:
        return out

    # --- Ψ M-step with BC1 analytic correction (Breslow-Lin 1995) ---
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
    mu_f = torch.sigmoid(eta_f)
    w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n
    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q_bm)
    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv_f = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4
    bg_outer_mat = torch.einsum('bmq,bmr->bmqr', blups, blups)
    Psi_lap_raw = (bg_outer_mat + Hg_inv_f).sum(dim=1) / G[:, None, None]

    # BC1: E[b_g|y,Ψ] ≈ b̂_g + f'''/(2H²) → ΔΨ_{jj} = (1/G)Σ_g b̂_{gj}·T3_{gj}·[H_g^{-1}]_{jj}²
    # T3_{gj} = -Σ_i z_{gij}³·μ_i(1-μ_i)(1-2μ_i)  (third log-likelihood derivative)
    mu_skew = mu_f * (1.0 - mu_f) * (1.0 - 2.0 * mu_f) * mask_n
    T3_g = -torch.einsum('bmnq,bmn->bmq', Zm.pow(3), mu_skew)
    hg_inv_diag = Hg_inv_f.diagonal(dim1=-2, dim2=-1)
    bc1_diag = (blups * T3_g * hg_inv_diag.square() * mask_m[:, :, None]).sum(dim=1) / G[:, None]
    bc1_diag = bc1_diag.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    Psi_lap_new = _psdProject(Psi_lap_raw + torch.diag_embed(bc1_diag))
    Psi_lap_new = _psdClampEigenvalues(Psi_lap_new, _BERNOULLI_PSI_EIG_CAP)
    sigma_rfx_new = Psi_lap_new.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    out['blup_est'] = blups
    out['sigma_rfx_est'] = sigma_rfx_new
    out['Psi_lap'] = Psi_lap_new
    return out


def refineBernoulliNestedBeta(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor | None = None,
    tau_ffx: torch.Tensor | None = None,
    family_ffx: torch.Tensor | None = None,
    n_beta_steps: int = 12,
    n_inner: int = 4,
    n_final: int = 3,
    damping: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Refine β via nested Newton: re-converge b̂_g at each β step (INLA-style).

    At every outer β Newton step, runs n_inner inner Newton steps on b̂_g under the
    current β before computing the Laplace score and Schur-complement Hessian.  This
    ensures the gradient ∑_g X_g'(y_g − σ(Xβ + Zb̂_g)) uses the MAP b̂_g at each β,
    matching INLA's nested optimization.  Ψ is held fixed at the nAGQ estimate.

    P6 used 2 outer rounds of (8 β steps at fixed blups, 3 blup steps).  At large d
    the PQL starting point is far from optimum and block coordinate ascent from it
    doesn't converge in 2 rounds.  Nested optimization has no such failure mode.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or n_beta_steps <= 0:
        return stats

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype

    active_d_mask = normal_prec = is_student = zeros_d = None
    if nu_ffx is not None and tau_ffx is not None:
        active_d_mask = tau_ffx > 0
        zeros_d = tau_ffx.new_zeros(tau_ffx.shape)
        normal_prec = torch.where(active_d_mask, 1.0 / tau_ffx.clamp(min=1e-4).square(), zeros_d)
        is_student = (family_ffx == 1).unsqueeze(-1) if family_ffx is not None else None

    has_rfx = q > 0 and 'Psi_lap' in stats
    if has_rfx:
        Psi_inv = _pseudoInverse(stats['Psi_lap'])
        eye_q = torch.eye(q, device=device, dtype=dtype)
        eye_q_bm = eye_q.expand(B, m, q, q)
        active = mask_m.bool()
        mask4 = mask_m[:, :, None, None]
        G = mask_m.sum(dim=1).clamp(min=1.0)

    beta = stats['beta_est'].detach().clone()
    blups = stats['blup_est'].detach().clone()

    for _ in range(n_beta_steps):
        # Inner: converge b̂_g at current β (Ψ fixed)
        if has_rfx:
            for _ in range(n_inner):
                eta_in = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum(
                    'bmnq,bmq->bmn', Zm, blups
                )
                mu_in = torch.sigmoid(eta_in)
                w_in = (mu_in * (1.0 - mu_in)).clamp(min=1e-6) * mask_n
                score_g = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_in) * mask_n)
                score_g = score_g - torch.einsum('bqr,bmr->bmq', Psi_inv, blups)
                ZWZ_in = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_in, Zm)
                ZWZ_in_safe = torch.where(active[:, :, None, None], ZWZ_in, eye_q_bm)
                Hg_in = ZWZ_in_safe + Psi_inv[:, None]
                delta_b = _safeSolve(Hg_in + _adaptiveRidgeBm(Hg_in), score_g)
                blups = (blups + damping * delta_b) * mask_m[:, :, None]
                blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

        # Outer: β Newton step using exact Laplace score at converged b̂_g
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        if has_rfx:
            eta = eta + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        mu = torch.sigmoid(eta)
        w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n

        XtWX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)       # (B, d, d)
        score = torch.einsum('bmnd,bmn->bd', Xm, (ym - mu) * mask_n)  # (B, d)

        if normal_prec is not None:
            if is_student is not None:
                student_prec = torch.where(
                    active_d_mask,
                    6.0
                    / (5.0 * tau_ffx.clamp(min=1e-8).square() + (beta - nu_ffx).square()).clamp(
                        min=1e-8
                    ),
                    zeros_d,
                )
                prior_prec = torch.where(is_student, student_prec, normal_prec)
            else:
                prior_prec = normal_prec
            XtWX = XtWX + torch.diag_embed(prior_prec)
            score = score + prior_prec * (nu_ffx - beta)

        delta = _safeSolve(XtWX + _adaptiveRidge(XtWX), score)
        beta = (
            (beta + damping * delta).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        )

    # Final b̂_g convergence at the converged β
    if has_rfx and n_final > 0:
        for _ in range(n_final):
            eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum(
                'bmnq,bmq->bmn', Zm, blups
            )
            mu_f = torch.sigmoid(eta_f)
            w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n
            score_g = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_f) * mask_n)
            score_g = score_g - torch.einsum('bqr,bmr->bmq', Psi_inv, blups)
            ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)
            ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q_bm)
            Hg_f = ZWZ_f_safe + Psi_inv[:, None]
            delta_b = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), score_g)
            blups = (blups + damping * delta_b) * mask_m[:, :, None]
            blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        # Reassign final w_f for the M-step below
        eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        mu_f = torch.sigmoid(eta_f)
        w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n

    out = dict(stats)
    out['beta_est'] = beta

    if not has_rfx:
        return out

    # M-step + BC1 analytic correction (same as P6)
    if n_final == 0:
        eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        mu_f = torch.sigmoid(eta_f)
        w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n

    ZWZ_mstep = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)
    ZWZ_mstep_safe = torch.where(active[:, :, None, None], ZWZ_mstep, eye_q_bm)
    Hg_mstep = ZWZ_mstep_safe + Psi_inv[:, None]
    Hg_inv_mstep = _safeSolve(Hg_mstep + _adaptiveRidgeBm(Hg_mstep), eye_q_bm) * mask4
    bg_outer = torch.einsum('bmq,bmr->bmqr', blups, blups)
    Psi_lap_raw = (bg_outer + Hg_inv_mstep).sum(dim=1) / G[:, None, None]

    mu_skew = mu_f * (1.0 - mu_f) * (1.0 - 2.0 * mu_f) * mask_n
    T3_g = -torch.einsum('bmnq,bmn->bmq', Zm.pow(3), mu_skew)
    hg_inv_diag = Hg_inv_mstep.diagonal(dim1=-2, dim2=-1)
    bc1_diag = (blups * T3_g * hg_inv_diag.square() * mask_m[:, :, None]).sum(dim=1) / G[:, None]
    bc1_diag = bc1_diag.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    Psi_lap_new = _psdProject(Psi_lap_raw + torch.diag_embed(bc1_diag))
    Psi_lap_new = _psdClampEigenvalues(Psi_lap_new, _BERNOULLI_PSI_EIG_CAP)
    sigma_rfx_new = Psi_lap_new.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    out['blup_est'] = blups
    out['sigma_rfx_est'] = sigma_rfx_new
    out['Psi_lap'] = Psi_lap_new
    return out


def _bernoulliLaplaceModeDiag(
    beta: torch.Tensor,
    log_sigma_rfx: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    mask_q: torch.Tensor | None,
    n_inner: int,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate b_g MAP modes and Hessians for diagonal-Ψ Bernoulli GLMMs."""
    B, m, _, q = Zm.shape
    device, dtype = Zm.device, Zm.dtype
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )
    active = mask_m.bool()
    Z_eff = Zm * active_q[:, None, None, :].to(dtype)
    prec = torch.where(active_q, torch.exp(-2.0 * log_sigma_rfx).clamp(max=1e8), 1.0)
    eye_q = torch.eye(q, device=device, dtype=dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)

    blups = Zm.new_zeros(B, m, q)
    for _ in range(n_inner):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
        mu = torch.sigmoid(eta)
        w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n
        score_g = torch.einsum('bmnq,bmn->bmq', Z_eff, (ym - mu) * mask_n)
        score_g = score_g - prec[:, None, :] * blups
        ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
        H = ZWZ + torch.diag_embed(prec)[:, None]
        H_safe = torch.where(active[:, :, None, None], H, eye_q_bm)
        delta = _safeSolve(H_safe + _adaptiveRidgeBm(H_safe), score_g)
        blups = (blups + damping * delta) * mask_m[:, :, None] * active_q[:, None, :].to(dtype)
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    mu = torch.sigmoid(eta)
    w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n
    ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    H = ZWZ + torch.diag_embed(prec)[:, None]
    H = torch.where(active[:, :, None, None], H, eye_q_bm)
    return blups, H, active_q


def _bernoulliLaplaceEbTargetDiag(
    beta: torch.Tensor,
    log_sigma_rfx: torch.Tensor,
    blups: torch.Tensor,
    H: torch.Tensor,
    active_q: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor | None,
    tau_ffx: torch.Tensor | None,
    family_ffx: torch.Tensor | None,
    tau_rfx: torch.Tensor | None,
    family_sigma_rfx: torch.Tensor | None,
    mask_d: torch.Tensor | None,
    sigma_log_jacobian: bool,
) -> torch.Tensor:
    """Laplace-approximated log posterior target for diagonal Bernoulli GLMMs."""
    B, _, _, d = Xm.shape
    q = Zm.shape[-1]
    dtype = Xm.dtype
    active_q_f = active_q.to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    ll = (ym * eta - F.softplus(eta)) * mask_n
    ll_g = ll.sum(dim=-1)

    sigma = log_sigma_rfx.exp().clamp(min=1e-8)
    log_prior_b = -0.5 * (
        math.log(2.0 * math.pi)
        + 2.0 * log_sigma_rfx[:, None, :]
        + blups.square() / sigma[:, None, :].square()
    )
    log_prior_b = (log_prior_b * active_q_f[:, None, :]).sum(dim=-1)

    sign, log_det_H = torch.linalg.slogdet(H)
    log_det_H = torch.where(sign > 0, log_det_H, log_det_H.new_zeros(()))
    q_count = active_q_f.sum(dim=-1)
    laplace_g = ll_g + log_prior_b + 0.5 * q_count[:, None] * math.log(2.0 * math.pi)
    laplace_g = laplace_g - 0.5 * log_det_H
    target = (laplace_g * mask_m).sum(dim=-1)

    if nu_ffx is not None and tau_ffx is not None and family_ffx is not None:
        if mask_d is None:
            mask_d_lp = torch.ones(B, 1, d, device=beta.device, dtype=dtype)
        else:
            mask_d_lp = mask_d[:, :d].to(device=beta.device, dtype=dtype).unsqueeze(1)
        target = target + logProbFfx(
            beta.unsqueeze(1),
            nu_ffx[:, :d].unsqueeze(1),
            tau_ffx[:, :d].clamp(min=1e-8).unsqueeze(1),
            family_ffx,
            mask_d_lp,
        ).squeeze(1)

    if tau_rfx is not None and family_sigma_rfx is not None:
        target = target + logProbSigma(
            sigma.unsqueeze(1),
            tau_rfx[:, :q].clamp(min=1e-8).unsqueeze(1),
            family_sigma_rfx,
            active_q_f.unsqueeze(1),
        ).squeeze(1)
    if sigma_log_jacobian:
        target = target + (log_sigma_rfx * active_q_f).sum(dim=-1)

    return target


def refineBernoulliLaplaceEb(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor | None = None,
    tau_ffx: torch.Tensor | None = None,
    family_ffx: torch.Tensor | None = None,
    tau_rfx: torch.Tensor | None = None,
    family_sigma_rfx: torch.Tensor | None = None,
    mask_d: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    n_steps: int = 12,
    n_inner: int = 4,
    n_final: int = 6,
    lr: float = 0.05,
    damping: float = 0.7,
    sigma_start: float = 0.03,
    sigma_max: float = 20.0,
    beta_start: str = 'stats',
    sigma_init: str = 'stats',
    sigma_log_jacobian: bool = True,
    accept_only_improved: bool = True,
    early_stop: bool = False,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 1e-4,
    blup_fallback_beta_jump: float | None = 1.0,
    beta_output_cap: float | None = None,
    beta_output_cap_trigger: float | None = None,
    sigma_prior_cap: float | None = None,
    sigma_prior_cap_min_d: int | None = None,
    recompute_blup_after_calibration: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Diagonal single-mode Laplace-EB refinement for Bernoulli GLMMs.

    It optimizes a true Bernoulli Laplace objective over β and diagonal σ_rfx with
    a σ continuation cap. By default β starts from the current analytical estimate
    while the effective σ starts tiny, so random effects cannot immediately absorb
    fixed-effect signal. The underlying σ parameter starts from the current
    analytical estimate by default while the effective σ is capped by the
    continuation schedule. By default the target includes the log-σ Jacobian,
    optimizing the hyperposterior in log scale rather than the zero-mode σ density.
    """
    q = Zm.shape[-1]
    d = Xm.shape[-1]
    if q == 0 or d == 0 or n_steps <= 0:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )

    if beta_start == 'prior' and nu_ffx is not None:
        beta_init = nu_ffx[:, :d].detach().clone()
    else:
        beta_init = stats['beta_est'][:, :d].detach().clone()
    beta = beta_init.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
    beta.requires_grad_(True)

    sigma0 = Zm.new_full((B, q), float(sigma_start)).clamp(min=1e-4, max=sigma_max)
    if sigma_init == 'stats' and 'sigma_rfx_est' in stats:
        sigma0 = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)
    elif sigma_init == 'prior' and tau_rfx is not None:
        sigma0 = tau_rfx[:, :q].detach().clamp(min=1e-4, max=sigma_max)
    log_sigma = sigma0.log().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([beta, log_sigma], lr=lr)
    min_log_sigma = math.log(1e-4)
    start_log_cap = math.log(max(sigma_start, 1e-4))
    final_log_cap = math.log(sigma_max)
    best_loss = float('inf')
    stale_steps = 0
    n_steps_run = 0

    with torch.enable_grad():
        for step in range(n_steps):
            frac = 1.0 if n_steps == 1 else float(step) / float(n_steps - 1)
            log_cap = start_log_cap + frac * (final_log_cap - start_log_cap)
            optimizer.zero_grad(set_to_none=True)
            log_sigma_step = log_sigma.clamp(min=min_log_sigma, max=log_cap)
            blups, H, active_q_step = _bernoulliLaplaceModeDiag(
                beta,
                log_sigma_step,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                mask_q,
                n_inner=n_inner,
                damping=damping,
            )
            target = _bernoulliLaplaceEbTargetDiag(
                beta,
                log_sigma_step,
                blups,
                H,
                active_q_step,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx,
                tau_ffx,
                family_ffx,
                tau_rfx,
                family_sigma_rfx,
                mask_d,
                sigma_log_jacobian,
            )
            loss = -target.sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([beta, log_sigma], max_norm=10.0)
            optimizer.step()
            n_steps_run = step + 1
            loss_value = float(loss.detach().item())
            if loss_value < best_loss - early_stop_min_delta:
                best_loss = loss_value
                stale_steps = 0
            else:
                stale_steps += 1
            with torch.no_grad():
                beta.clamp_(-20.0, 20.0)
                log_sigma.clamp_(min_log_sigma, final_log_cap)
                log_sigma.masked_fill_(~active_q, min_log_sigma)
            if (
                early_stop
                and step >= max(n_steps // 2, 1)
                and early_stop_patience > 0
                and stale_steps >= early_stop_patience
            ):
                break

    with torch.no_grad():
        beta_final = beta.detach()
        log_sigma_final = log_sigma.detach().clamp(min=min_log_sigma, max=final_log_cap)
        blups, H, _ = _bernoulliLaplaceModeDiag(
            beta_final,
            log_sigma_final,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            mask_q,
            n_inner=n_final,
            damping=damping,
        )
        eye_q = torch.eye(q, device=device, dtype=dtype).expand(H.shape[0], H.shape[1], q, q)
        H_inv = _safeSolve(H + _adaptiveRidgeBm(H), eye_q) * mask_m[:, :, None, None]
        sigma = log_sigma_final.exp()
        if 'sigma_rfx_est' in stats:
            sigma = torch.where(active_q, sigma, stats['sigma_rfx_est'][:, :q])
        Psi_lap = torch.diag_embed(sigma.square())
        blup_var = H_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
        blup_var = blup_var * mask_m[:, :, None] * active_q[:, None, :].to(dtype)

        accept = torch.ones(B, device=device, dtype=torch.bool)
        blup_fallback = torch.zeros(B, device=device, dtype=torch.bool)
        beta_output_capped = torch.zeros(B, device=device, dtype=torch.bool)
        sigma_prior_capped = torch.zeros(B, device=device, dtype=torch.bool)
        beta_jump = torch.full((B,), float('nan'), device=device, dtype=dtype)
        final_target = torch.full((B,), float('nan'), device=device, dtype=dtype)
        base_target = torch.full((B,), float('nan'), device=device, dtype=dtype)
        base_beta = stats['beta_est'][:, :d].detach() if 'beta_est' in stats else None
        if accept_only_improved and 'sigma_rfx_est' in stats:
            base_log_sigma = (
                stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max).log()
            )
            base_blups, base_H, base_active_q = _bernoulliLaplaceModeDiag(
                base_beta,
                base_log_sigma,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                mask_q,
                n_inner=n_final,
                damping=damping,
            )
            final_target = _bernoulliLaplaceEbTargetDiag(
                beta_final,
                log_sigma_final,
                blups,
                H,
                active_q,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx,
                tau_ffx,
                family_ffx,
                tau_rfx,
                family_sigma_rfx,
                mask_d,
                sigma_log_jacobian,
            )
            base_target = _bernoulliLaplaceEbTargetDiag(
                base_beta,
                base_log_sigma,
                base_blups,
                base_H,
                base_active_q,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx,
                tau_ffx,
                family_ffx,
                tau_rfx,
                family_sigma_rfx,
                mask_d,
                sigma_log_jacobian,
            )
            accept = final_target >= base_target - 1e-5
            beta_final = torch.where(accept[:, None], beta_final, base_beta)
            sigma = torch.where(accept[:, None], sigma, stats['sigma_rfx_est'][:, :q])
            blups = torch.where(accept[:, None, None], blups, stats['blup_est'][:, :, :q])
            if 'blup_var' in stats:
                blup_var = torch.where(accept[:, None, None], blup_var, stats['blup_var'][:, :, :q])
            Psi_lap = torch.where(
                accept[:, None, None], torch.diag_embed(sigma.square()), stats['Psi_lap'][:, :q, :q]
            )
        if blup_fallback_beta_jump is not None and base_beta is not None and 'blup_est' in stats:
            if mask_d is None:
                active_d = torch.ones(B, d, device=device, dtype=torch.bool)
            else:
                active_d = mask_d[:, :d].to(device=device).bool()
            d_count = active_d.to(dtype).sum(dim=1).clamp(min=1.0)
            beta_diff2 = (beta_final - base_beta).square() * active_d.to(dtype)
            beta_jump = (beta_diff2.sum(dim=1) / d_count).sqrt()
            blup_fallback = accept & (beta_jump >= float(blup_fallback_beta_jump))
            blups = torch.where(blup_fallback[:, None, None], stats['blup_est'][:, :, :q], blups)
            if 'blup_var' in stats:
                blup_var = torch.where(
                    blup_fallback[:, None, None], stats['blup_var'][:, :, :q], blup_var
                )

        if beta_output_cap is not None:
            beta_capped = beta_final.clamp(-float(beta_output_cap), float(beta_output_cap))
            if beta_output_cap_trigger is None:
                use_cap = torch.ones(B, device=device, dtype=torch.bool)
            else:
                if mask_d is None:
                    active_d = torch.ones(B, d, device=device, dtype=torch.bool)
                else:
                    active_d = mask_d[:, :d].to(device=device).bool()
                max_abs_beta = beta_final.abs().masked_fill(~active_d, 0.0).amax(dim=1)
                use_cap = max_abs_beta > float(beta_output_cap_trigger)
            beta_output_capped = use_cap
            beta_final = torch.where(use_cap[:, None], beta_capped, beta_final)

        if sigma_prior_cap is not None and tau_rfx is not None:
            sigma_cap = float(sigma_prior_cap) * tau_rfx[:, :q].to(device=device).clamp(min=1e-4)
            sigma_capped = torch.minimum(sigma, sigma_cap)
            cap_active = active_q & (sigma_capped < sigma)
            if sigma_prior_cap_min_d is not None:
                if mask_d is None:
                    d_count = torch.full((B,), d, device=device, dtype=torch.long)
                else:
                    d_count = mask_d[:, :d].to(device=device).bool().sum(dim=1)
                cap_active = cap_active & (d_count >= int(sigma_prior_cap_min_d))[:, None]
            sigma_prior_capped = cap_active.any(dim=1)
            sigma = torch.where(cap_active, sigma_capped, sigma)
            Psi_lap = torch.diag_embed(sigma.square())
            if recompute_blup_after_calibration and sigma_prior_capped.any():
                blups_new, H_new, _ = _bernoulliLaplaceModeDiag(
                    beta_final,
                    sigma.clamp(min=1e-4, max=sigma_max).log(),
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    mask_q,
                    n_inner=n_final,
                    damping=damping,
                )
                H_inv_new = (
                    _safeSolve(H_new + _adaptiveRidgeBm(H_new), eye_q) * mask_m[:, :, None, None]
                )
                blup_var_new = H_inv_new.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
                blup_var_new = blup_var_new * mask_m[:, :, None] * active_q[:, None, :].to(dtype)
                blups = torch.where(sigma_prior_capped[:, None, None], blups_new, blups)
                blup_var = torch.where(sigma_prior_capped[:, None, None], blup_var_new, blup_var)

    out = dict(stats)
    out['beta_est'] = beta_final
    out['sigma_rfx_est'] = sigma
    out['blup_est'] = blups.detach()
    out['blup_var'] = blup_var.detach()
    out['Psi_lap'] = _psdClampEigenvalues(Psi_lap, _BERNOULLI_PSI_EIG_CAP)
    if return_diagnostics:
        out['laplace_eb_accept'] = accept.to(dtype)
        out['laplace_eb_steps'] = torch.full((B,), float(n_steps_run), device=device, dtype=dtype)
        out['laplace_eb_target'] = final_target
        out['laplace_eb_base_target'] = base_target
        out['laplace_eb_blup_fallback'] = blup_fallback.to(dtype)
        out['laplace_eb_beta_jump'] = beta_jump
        out['laplace_eb_beta_output_capped'] = beta_output_capped.to(dtype)
        out['laplace_eb_sigma_prior_capped'] = sigma_prior_capped.to(dtype)
    return out


def _ghProductGrid(
    k_vals: list[int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian-product Gauss-Hermite grid.

    Returns z_nodes (K, q) and log_w (K,) where K = prod(k_vals).
    The +‖z‖² correction for each point cancels the implicit GH weight in
    the substitution ∫ g(x)dx ≈ Σ_j w_j g(z_j) e^{z_j²}.
    """
    z_1d, logw_1d = [], []
    for k in k_vals:
        z_np, w_np = np.polynomial.hermite.hermgauss(k)
        z_1d.append(torch.tensor(z_np, dtype=dtype, device=device))
        logw_1d.append(torch.tensor(np.log(w_np), dtype=dtype, device=device))
    if len(k_vals) == 1:
        return z_1d[0].unsqueeze(-1), logw_1d[0]
    grids_z = torch.meshgrid(*z_1d, indexing='ij')
    grids_w = torch.meshgrid(*logw_1d, indexing='ij')
    z_nodes = torch.stack([g.reshape(-1) for g in grids_z], dim=-1)
    log_w = sum(g.reshape(-1) for g in grids_w)
    return z_nodes, log_w


def refineBernoulliNagqSrfx(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    mask_q: torch.Tensor | None = None,
    n_steps: int = 10,
    n_newton: int = 3,
    k: int = 7,
    lr: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Refine σ_rfx for Bernoulli GLMMs via nAGQ LML gradient.

    For active_q == 1: scalar adaptive GH quadrature (k points per dataset).
    For 2 <= active_q <= 5: Cartesian product GH grid (k_mv^q_act points) where
    k_mv is chosen per q_act to keep total nodes tractable (25/125/81/243).
    Datasets with active_q == 0 or active_q > 5 are returned unchanged.

    The +‖z‖² term in the logsumexp cancels the implicit GH weight:
        LML_g = logsumexp_j(log w_j + ℓ_{g,j} + ‖z_j‖²)
                + ½ q_act log 2 − ½ log|H_g|
    where b_{g,j} = b̂_g + √2·L_g·z_j, L_g = chol(H_g^{-1}),
    H_g = ZWZ_g + Ψ^{-1}, and ℓ_{g,j} = log p(y_g|β,b_{g,j}) + log p(b_{g,j}|Ψ).
    ZWZ_g is fixed at the PQL mode throughout optimization.

    After nAGQ gradient steps, b̂_g is recomputed via n_newton Newton steps under
    the refined Ψ for all eligible datasets.
    """
    q = Zm.shape[-1]
    if q == 0 or 'Psi_lap' not in stats:
        return stats

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype
    n_max = Zm.shape[2]

    if mask_q is not None:
        active_q_count = mask_q[:, :q].long().sum(dim=-1)  # (B,)
    else:
        active_q_count = torch.full((B,), q, device=device, dtype=torch.long)

    nagq1_eligible = active_q_count == 1  # scalar path
    mv_eligible = (active_q_count >= 2) & (active_q_count <= 5)  # product GH path
    any_eligible = nagq1_eligible | mv_eligible
    if not any_eligible.any():
        return stats

    beta = stats['beta_est'].detach()
    eta_fix = torch.einsum('bmnd,bd->bmn', Xm, beta).detach()  # (B, m, n_max)
    Psi_lap_new = stats['Psi_lap'].detach().clone()

    # ---- q == 1: scalar nAGQ ----
    if nagq1_eligible.any():
        z_np, w_np = np.polynomial.hermite.hermgauss(k)
        z_nodes = torch.tensor(z_np, dtype=dtype, device=device)  # (k,)
        log_w = torch.tensor(np.log(w_np), dtype=dtype, device=device)  # (k,)

        if mask_q is not None:
            q_idx = mask_q[:, :q].long().argmax(dim=-1)  # (B,) first active dim
        else:
            q_idx = torch.zeros(B, device=device, dtype=torch.long)
        b_arange = torch.arange(B, device=device)

        z_col = Zm.gather(3, q_idx[:, None, None, None].expand(B, m, n_max, 1)).squeeze(-1)
        b_g0 = (
            stats['blup_est'].gather(2, q_idx[:, None, None].expand(B, m, 1)).squeeze(-1).detach()
        )
        log_s2 = stats['Psi_lap'][b_arange, q_idx, q_idx].clamp(min=1e-8).log().detach().clone()
        log_s2.requires_grad_(True)

        with torch.no_grad():
            mu0 = torch.sigmoid((eta_fix + z_col * b_g0[:, :, None]) * mask_n)
            ZWZ_g1 = (z_col.square() * (mu0 * (1.0 - mu0)).clamp(min=1e-6) * mask_n).sum(-1)

        elig1 = nagq1_eligible.to(dtype=dtype)
        optimizer1 = torch.optim.Adam([log_s2], lr=lr)

        with torch.enable_grad():
            for _ in range(n_steps):
                optimizer1.zero_grad(set_to_none=True)
                s2 = log_s2.exp()
                H_g = ZWZ_g1 + 1.0 / s2[:, None].clamp(min=1e-8)  # (B, m)
                sg = 1.0 / H_g.clamp(min=1e-8).sqrt()
                b_gj = b_g0[:, :, None] + math.sqrt(2.0) * sg[:, :, None] * z_nodes[None, None, :]
                eta_gj = eta_fix[:, :, :, None] + z_col[:, :, :, None] * b_gj[:, :, None, :]
                mn4 = mask_n[:, :, :, None]
                ll_gj = (ym[:, :, :, None] * eta_gj * mn4 - F.softplus(eta_gj) * mn4).sum(2)
                lp_gj = (
                    -0.5 * math.log(2.0 * math.pi)
                    - 0.5 * log_s2[:, None, None]
                    - b_gj.square() / (2.0 * s2[:, None, None].clamp(min=1e-8))
                )
                lml_g = (
                    torch.logsumexp(
                        log_w[None, None, :] + ll_gj + lp_gj + z_nodes[None, None, :].square(),
                        dim=-1,
                    )
                    + 0.5 * math.log(2.0)
                    - 0.5 * H_g.clamp(min=1e-8).log()
                )
                lml = ((lml_g * mask_m).sum(-1) * elig1).sum()
                if not torch.isfinite(lml):
                    break
                (-lml).backward()
                torch.nn.utils.clip_grad_norm_([log_s2], max_norm=5.0)
                optimizer1.step()
                with torch.no_grad():
                    log_s2.clamp_(math.log(1e-4), math.log(20.0))

        s2_final = log_s2.detach().exp()
        for b in nagq1_eligible.nonzero(as_tuple=True)[0].tolist():
            Psi_lap_new[b, int(q_idx[b]), int(q_idx[b])] = s2_final[b]

    # ---- q >= 2: product GH ----
    _K_MV = {2: 5, 3: 5, 4: 3, 5: 3}
    for q_act in range(2, 6):
        elig_bs = (active_q_count == q_act).nonzero(as_tuple=True)[0].tolist()
        if not elig_bs:
            continue
        n_elig = len(elig_bs)

        k_mv = _K_MV[q_act]
        z_grid, log_w_grid = _ghProductGrid([k_mv] * q_act, dtype, device)
        # z_grid: (K, q_act), log_w_grid: (K,)
        lz2 = z_grid.pow(2).sum(dim=-1)  # (K,) — ‖z_j‖² correction

        # Per-item active column indices
        act_idx = []
        for b in elig_bs:
            if mask_q is not None:
                ai = mask_q[b, :q].nonzero(as_tuple=True)[0]
            else:
                ai = torch.arange(q_act, device=device)
            act_idx.append(ai)

        # Stack across eligible items  (n_elig, m, n_max, q_act), etc.
        z_cols_e = torch.stack([Zm[b, :, :, act_idx[i]] for i, b in enumerate(elig_bs)])
        b_g0_e = torch.stack(
            [stats['blup_est'][b, :, act_idx[i]] for i, b in enumerate(elig_bs)]
        ).detach()
        eta_fix_e = eta_fix[elig_bs]  # (n_elig, m, n_max)
        mask_m_e = mask_m[elig_bs].float()
        mask_n_e = mask_n[elig_bs]
        ym_e = ym[elig_bs]

        with torch.no_grad():
            mu0_e = torch.sigmoid(eta_fix_e + torch.einsum('emnq,emq->emn', z_cols_e, b_g0_e))
            w0_e = (mu0_e * (1.0 - mu0_e)).clamp(min=1e-6) * mask_n_e
            ZWZ_g_e = torch.einsum('emnq,emn,emnr->emqr', z_cols_e, w0_e, z_cols_e)

        log_s2_e_init = torch.stack(
            [
                Psi_lap_new[b][act_idx[i], act_idx[i]].clamp(min=1e-8).log()
                for i, b in enumerate(elig_bs)
            ]
        ).detach()  # (n_elig, q_act) — kept as fallback if optimization diverges
        log_s2_e = log_s2_e_init.clone().requires_grad_(True)

        eye_q_act = torch.eye(q_act, device=device, dtype=dtype)
        optimizer_mv = torch.optim.Adam([log_s2_e], lr=lr)

        with torch.enable_grad():
            for _ in range(n_steps):
                optimizer_mv.zero_grad(set_to_none=True)
                s2_e = log_s2_e.exp()  # (n_elig, q_act)

                # H_g = ZWZ_g + diag(1/s2)  (n_elig, m, q_act, q_act)
                H_g_e = (
                    ZWZ_g_e
                    + torch.diag_embed(1.0 / s2_e.clamp(min=1e-8))[:, None]
                    + 1e-6 * eye_q_act[None, None]
                )
                chol_Hg = torch.linalg.cholesky(H_g_e)
                log_det_Hg = 2.0 * chol_Hg.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (n_elig, m)

                # L_g = chol(H_g^{-1}): adaptive quadrature spread
                H_g_inv = (
                    torch.cholesky_solve(
                        eye_q_act[None, None].expand(n_elig, m, q_act, q_act), chol_Hg
                    )
                    + 1e-6 * eye_q_act[None, None]
                )
                L_g = torch.linalg.cholesky(H_g_inv)  # (n_elig, m, q_act, q_act)

                # b_gj = b̂_g + √2·L_g·z_j  (n_elig, m, q_act, K)
                b_gj_e = b_g0_e[:, :, :, None] + math.sqrt(2.0) * torch.einsum(
                    'emqr,Kr->emqK', L_g, z_grid
                )

                # eta_gj = eta_fix + z_cols @ b_gj  (n_elig, m, n_max, K)
                eta_gj_e = eta_fix_e[:, :, :, None] + torch.einsum(
                    'emnq,emqK->emnK', z_cols_e, b_gj_e
                )

                mn4 = mask_n_e[:, :, :, None]
                ll_gj_e = (ym_e[:, :, :, None] * eta_gj_e - F.softplus(eta_gj_e)) * mn4
                ll_gj_e = ll_gj_e.sum(dim=2)  # (n_elig, m, K)

                lp_gj_e = (
                    -0.5 * q_act * math.log(2.0 * math.pi)
                    - 0.5 * log_s2_e.sum(dim=-1)[:, None, None]
                    - 0.5 * (b_gj_e.pow(2) / s2_e[:, None, :, None].clamp(min=1e-8)).sum(dim=2)
                )  # (n_elig, m, K)

                lml_g_e = (
                    torch.logsumexp(
                        log_w_grid[None, None, :] + ll_gj_e + lp_gj_e + lz2[None, None, :],
                        dim=-1,
                    )
                    + 0.5 * q_act * math.log(2.0)
                    - 0.5 * log_det_Hg
                )  # (n_elig, m)

                lml_mv = (lml_g_e * mask_m_e).sum()
                if not torch.isfinite(lml_mv):
                    break
                (-lml_mv).backward()
                if log_s2_e.grad is not None and not log_s2_e.grad.isfinite().all():
                    break
                torch.nn.utils.clip_grad_norm_([log_s2_e], max_norm=5.0)
                optimizer_mv.step()
                with torch.no_grad():
                    log_s2_e.clamp_(math.log(1e-4), math.log(20.0))

        # Fall back to initial values wherever optimization diverged
        log_s2_safe = torch.where(log_s2_e.detach().isfinite(), log_s2_e.detach(), log_s2_e_init)
        s2_final_e = log_s2_safe.exp()
        for i, b in enumerate(elig_bs):
            for qi_loc, qi_glob in enumerate(act_idx[i].tolist()):
                Psi_lap_new[b, qi_glob, qi_glob] = s2_final_e[i, qi_loc]

    Psi_lap_new = _psdClampEigenvalues(Psi_lap_new, _BERNOULLI_PSI_EIG_CAP)

    # Newton BLUP refresh under refined Ψ (runs for all B, restored below for ineligible)
    blups = stats['blup_est'].detach().clone()
    Psi_inv_new = _pseudoInverse(Psi_lap_new)
    eye_q = torch.eye(q, device=device, dtype=dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)
    active = mask_m.bool()

    for _ in range(n_newton):
        eta_b = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        mu_b = torch.sigmoid(eta_b)
        w_b = (mu_b * (1.0 - mu_b)).clamp(min=1e-6) * mask_n
        score_g = torch.einsum('bmnq,bmn->bmq', Zm, (ym - mu_b) * mask_n) - torch.einsum(
            'bqr,bmr->bmq', Psi_inv_new, blups
        )
        ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_b, Zm)
        Hg = torch.where(active[:, :, None, None], ZWZ, eye_q_bm) + Psi_inv_new[:, None]
        blups = (blups + _safeSolve(Hg + _adaptiveRidgeBm(Hg), score_g)) * mask_m[:, :, None]
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    for b in (~any_eligible).nonzero(as_tuple=True)[0].tolist():
        blups[b] = stats['blup_est'][b]

    sigma_rfx_new = Psi_lap_new.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()
    out = dict(stats)
    out['blup_est'] = blups
    out['sigma_rfx_est'] = sigma_rfx_new
    out['Psi_lap'] = Psi_lap_new
    return out
