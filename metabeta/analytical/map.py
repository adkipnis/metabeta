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
    'refineBernoulliMapBeta',
    'refineBernoulliMapSrfx',
    'refineBernoulliNagqSrfx',
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

    beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
    active_d_count = (XtX.diagonal(dim1=-2, dim2=-1).abs() > 1e-8).sum(dim=-1)
    alpha = torch.where(
        active_d_count <= 8,
        se2.new_full(se2.shape, beta_alpha_low),
        se2.new_full(se2.shape, beta_alpha_high),
    )
    beta_for_blup = ((1.0 - alpha[:, None]) * gls.beta + alpha[:, None] * beta_ols).nan_to_num(
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
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
    out['beta_est'] = gls.beta
    out['sigma_rfx_est'] = sigma_rfx
    out['blup_est'] = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
    out['blup_var'] = blup_var
    out['Psi'] = Psi
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
            )
        out['Psi'] = _replacePsiDiag(stats['Psi'], sigma_rfx, mask_q)
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

    # --- Ψ M-step: Ψ = (1/G) Σ_g (b̂_g b̂_g' + H_g^{-1}) with final b̂_g ---
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
    mu_f = torch.sigmoid(eta_f)
    w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n
    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q_bm)
    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv_f = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4
    bg_outer_mat = torch.einsum('bmq,bmr->bmqr', blups, blups)
    Psi_lap_new = _psdProject((bg_outer_mat + Hg_inv_f).sum(dim=1) / G[:, None, None])
    Psi_lap_new = _psdClampEigenvalues(Psi_lap_new, _BERNOULLI_PSI_EIG_CAP)
    sigma_rfx_new = Psi_lap_new.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    out['blup_est'] = blups
    out['sigma_rfx_est'] = sigma_rfx_new
    out['Psi_lap'] = Psi_lap_new
    return out


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
    """Refine σ_rfx for Bernoulli GLMMs with exactly one active RE via nAGQ LML gradient.

    For datasets with active_q == 1 (scalar random effect), replaces the PQL Laplace
    M-step σ_rfx with a gradient-step estimate from k-point adaptive Gauss-Hermite
    quadrature of the marginal log-likelihood.  Datasets with active_q > 1 are returned
    unchanged; the function gates on active_q.sum() == 1 per batch item.

    After the nAGQ gradient step, b̂_g is recomputed via n_newton Newton steps under
    the refined Ψ to produce consistent BLUPs.

    The +z_j² term in the logsumexp cancels the implicit Gaussian weight in standard GH:
        LML_g = logsumexp_j(log w_j + ℓ_{g,j} + z_j²) + log(√2 · σ_g)
    where σ_g = H_g^{-1/2}, H_g = ZWZ_g + σ^{-2}, b_{g,j} = b̂_g + √2·σ_g·z_j,
    and ℓ_{g,j} = log p(y_g | β, b_{g,j}) + log p(b_{g,j} | σ²).
    """
    q = Zm.shape[-1]
    if q == 0 or 'Psi_lap' not in stats:
        return stats

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype

    # Gate: apply only to datasets with exactly 1 active RE dimension
    if mask_q is not None:
        active_q_count = mask_q[:, :q].long().sum(dim=-1)
    else:
        active_q_count = torch.full((B,), q, device=device, dtype=torch.long)
    nagq_eligible = active_q_count == 1  # (B,) bool
    if not nagq_eligible.any():
        return stats

    # GH nodes/weights: ∫ f(x) e^{-x²} dx ≈ Σ_j w_j f(z_j)
    z_np, w_np = np.polynomial.hermite.hermgauss(k)
    z_nodes = torch.tensor(z_np, dtype=dtype, device=device)       # (k,)
    log_w = torch.tensor(np.log(w_np), dtype=dtype, device=device)   # (k,)

    # First active q index per dataset
    if mask_q is not None:
        q_idx = mask_q[:, :q].long().argmax(dim=-1)  # (B,)
    else:
        q_idx = torch.zeros(B, device=device, dtype=torch.long)

    n_max = Zm.shape[2]
    b_arange = torch.arange(B, device=device)

    # Gather the single active RE z-column  (B, m, n_max)
    z_col = Zm.gather(3, q_idx[:, None, None, None].expand(B, m, n_max, 1)).squeeze(-1)

    # PQL BLUPs for the active dim — fixed quadrature centers  (B, m)
    b_g0 = stats['blup_est'].gather(2, q_idx[:, None, None].expand(B, m, 1)).squeeze(-1).detach()

    # Initial log σ² from active Psi_lap diagonal entry
    log_s2 = stats['Psi_lap'][b_arange, q_idx, q_idx].clamp(min=1e-8).log().detach().clone()
    log_s2.requires_grad_(True)

    beta = stats['beta_est'].detach()
    eta_fix = torch.einsum('bmnd,bd->bmn', Xm, beta).detach()  # (B, m, n)

    # ZWZ_g scalar per group (fixed at PQL mode)
    with torch.no_grad():
        mu0 = torch.sigmoid((eta_fix + z_col * b_g0[:, :, None]) * mask_n)
        ZWZ_g = (z_col.square() * (mu0 * (1.0 - mu0)).clamp(min=1e-6) * mask_n).sum(-1)

    elig = nagq_eligible.to(dtype=dtype)
    optimizer = torch.optim.Adam([log_s2], lr=lr)

    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            s2 = log_s2.exp()

            # H_g = ZWZ_g + Ψ^{-1}; curvature σ_g = H_g^{-1/2}
            H_g = ZWZ_g + 1.0 / s2[:, None].clamp(min=1e-8)  # (B, m)
            sg = 1.0 / H_g.clamp(min=1e-8).sqrt()

            # Quadrature points b_{g,j} = b̂_g + √2·σ_g·z_j  (B, m, k)
            b_gj = b_g0[:, :, None] + math.sqrt(2.0) * sg[:, :, None] * z_nodes[None, None, :]

            # Log-likelihood sum over observations  (B, m, k)
            eta_gj = eta_fix[:, :, :, None] + z_col[:, :, :, None] * b_gj[:, :, None, :]
            mn4 = mask_n[:, :, :, None]
            ll_gj = (ym[:, :, :, None] * eta_gj * mn4 - F.softplus(eta_gj) * mn4).sum(2)

            # Log-prior N(0, σ²)  (B, m, k)
            lp_gj = (
                -0.5 * math.log(2.0 * math.pi)
                - 0.5 * log_s2[:, None, None]
                - b_gj.square() / (2.0 * s2[:, None, None].clamp(min=1e-8))
            )

            # nAGQ LML per group; +z_j² cancels the implicit GH weight  (B, m)
            lml_g = (
                torch.logsumexp(
                    log_w[None, None, :] + ll_gj + lp_gj + z_nodes[None, None, :].square(),
                    dim=-1,
                )
                + 0.5 * math.log(2.0)
                - 0.5 * H_g.clamp(min=1e-8).log()
            )

            lml = ((lml_g * mask_m).sum(-1) * elig).sum()
            if not torch.isfinite(lml):
                break
            (-lml).backward()
            torch.nn.utils.clip_grad_norm_([log_s2], max_norm=5.0)
            optimizer.step()
            with torch.no_grad():
                log_s2.clamp_(math.log(1e-4), math.log(20.0))

    # Update Psi_lap active diagonal for eligible datasets
    s2_final = log_s2.detach().exp()
    Psi_lap_new = stats['Psi_lap'].detach().clone()
    for b in nagq_eligible.nonzero(as_tuple=True)[0].tolist():
        qi = int(q_idx[b])
        Psi_lap_new[b, qi, qi] = s2_final[b]
    Psi_lap_new = _psdClampEigenvalues(Psi_lap_new, _BERNOULLI_PSI_EIG_CAP)

    # Recompute b̂_g via Newton under refined Ψ
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

    # Restore ineligible datasets to original BLUPs
    for b in (~nagq_eligible).nonzero(as_tuple=True)[0].tolist():
        blups[b] = stats['blup_est'][b]

    sigma_rfx_new = Psi_lap_new.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()
    out = dict(stats)
    out['blup_est'] = blups
    out['sigma_rfx_est'] = sigma_rfx_new
    out['Psi_lap'] = Psi_lap_new
    return out


def refineBernoulliMapSrfx(
    stats: dict[str, torch.Tensor],
    G: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    mask_q: torch.Tensor | None = None,
    n_steps: int = 20,
    lr: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Refine sigma_rfx for Bernoulli GLMMs via fixed-point Laplace MAP.

    Uses PQL outputs (Psi_lap, mean_Hg_inv, sigma_rfx_est) — no data re-access
    needed.  The M-step identity Ψ_lap = (1/G)(Σ b̂_g b̂_g' + Σ H_g^{-1}) gives
    Σ_g b̂_gj² = G·(Ψ_lap_jj − mean_Hg_inv_jj), which serves as the sufficient
    statistic for MAP optimization of log σ_j under the rfx prior.

    Inactive rfx dimensions (tau_rfx == 0) are excluded from the objective and
    kept at their Laplace M-step values in the output.
    """
    q = tau_rfx.shape[-1]
    if q == 0 or n_steps <= 0:
        return stats

    Psi_lap = stats['Psi_lap'][..., :q, :q]
    mean_Hg_inv = stats['mean_Hg_inv'][..., :q, :q]
    sigma_lap = stats['sigma_rfx_est'][..., :q].clamp(min=1e-4)

    psi_diag = Psi_lap.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    h_diag = mean_Hg_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    sum_bhat_sq = (G[:, None] * (psi_diag - h_diag)).clamp(min=0.0)

    # Active rfx mask: exclude inactive dims (tau==0) from objective and prior.
    active_q = tau_rfx[..., :q] > 0
    if mask_q is not None:
        active_q = active_q & mask_q[..., :q].bool()
    active_float = active_q.float()
    mask_lp = active_float.unsqueeze(1)
    tau_lp = tau_rfx[..., :q].clamp(min=1e-4).unsqueeze(1)

    log_sigma = sigma_lap.log().detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([log_sigma], lr=lr)

    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            sigma = log_sigma.exp()
            sigma2 = sigma.square().clamp(min=1e-12)
            ll = -0.5 * ((sum_bhat_sq / sigma2 + G[:, None] * sigma2.log()) * active_float).sum()
            lp = logProbSigma(sigma.unsqueeze(1), tau_lp, family_sigma_rfx, mask_lp).sum()
            loss = -(ll + lp)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([log_sigma], max_norm=5.0)
            optimizer.step()
            with torch.no_grad():
                log_sigma.clamp_(math.log(1e-4), math.log(20.0))

    sigma_rfx_map = log_sigma.detach().exp()
    sigma_rfx_map = torch.where(active_q, sigma_rfx_map, sigma_lap)
    out = dict(stats)
    out['sigma_rfx_est'] = sigma_rfx_map
    return out
