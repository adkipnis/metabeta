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
    'refineBernoulliNagqSrfx',
    'refineBernoulliNestedBeta',
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

    blups = stats['blup_est'].detach().clone()
    beta = stats['beta_est'].detach().clone()

    for _ in range(n_beta_steps):
        # Inner: converge b̂_g at current β (Ψ fixed)
        if has_rfx:
            for _ in range(n_inner):
                eta_in = (
                    torch.einsum('bmnd,bd->bmn', Xm, beta)
                    + torch.einsum('bmnq,bmq->bmn', Zm, blups)
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
        beta = (beta + damping * delta).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(
            -50.0, 50.0
        )

    # Final b̂_g convergence at the converged β
    if has_rfx and n_final > 0:
        for _ in range(n_final):
            eta_f = (
                torch.einsum('bmnd,bd->bmn', Xm, beta)
                + torch.einsum('bmnq,bmq->bmn', Zm, blups)
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
        eta_f = (
            torch.einsum('bmnd,bd->bmn', Xm, beta)
            + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        )
        mu_f = torch.sigmoid(eta_f)
        w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n

    out = dict(stats)
    out['beta_est'] = beta

    if not has_rfx:
        return out

    # M-step + BC1 analytic correction (same as P6)
    if n_final == 0:
        eta_f = (
            torch.einsum('bmnd,bd->bmn', Xm, beta)
            + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        )
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
