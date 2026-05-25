"""MAP/EB refinements for Gaussian GLMMs."""

import math

import torch
from torch.nn import functional as F

from metabeta.analytical.linalg import (
    _adaptiveRidge,
    _adaptiveRidgeBm,
    _safeSolve,
)
from metabeta.analytical.lmm.lmm import _normalGlsAndBlups
from metabeta.utils.families import logProbFfx, logProbSigma

__all__ = [
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
    return_condition: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Approximate hyperparameter-averaged β over a small diagonal-σ grid."""
    q = Zm.shape[-1]
    d = Xm.shape[-1]
    if mask_q is not None:
        Zm = Zm * mask_q[..., :q].to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :]

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype
    clean_scales = [float(scale) for scale in scales if float(scale) > 0.0]
    if not any(abs(scale - 1.0) <= 1e-8 for scale in clean_scales):
        clean_scales.append(1.0)
    clean_scales = sorted(set(clean_scales))

    candidates = [
        (sigma_rfx_base * float(scale)).clamp(min=1e-4, max=20.0) for scale in clean_scales
    ]
    base_idx = next(i for i, scale in enumerate(clean_scales) if abs(scale - 1.0) <= 1e-8)
    sigma_grid = torch.stack(candidates, dim=1)
    S = sigma_grid.shape[1]
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

    A_data = (XtX[:, None] - correction_XX) / se2[:, None, None, None]
    A = A_data
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
    if return_condition:
        beta_cond = _normalBetaPrecisionCondition(A_data[:, base_idx], mask_d)
        return beta_avg, beta_cond
    return beta_avg


def _normalBetaPrecisionCondition(
    precision: torch.Tensor,
    mask_d: torch.Tensor | None,
) -> torch.Tensor:
    """Condition number of the data precision matrix; large values indicate weak β identification."""
    B, d, _ = precision.shape
    device, dtype = precision.device, precision.dtype
    precision = 0.5 * (precision + precision.mT)
    if mask_d is None:
        active_d = torch.ones((B, d), device=device, dtype=torch.bool)
    else:
        active_d = mask_d[..., :d].to(device=device).bool()
    eye = torch.eye(d, device=device, dtype=dtype)
    active_outer = active_d[:, :, None] & active_d[:, None, :]
    precision = torch.where(active_outer, precision, precision.new_zeros(()))
    precision = precision + eye[None] * (~active_d).to(dtype)[:, None, :]
    vals = torch.linalg.eigvalsh(precision).clamp(min=0.0)
    max_val = vals.amax(dim=-1).clamp(min=1e-12)
    positive = vals > (max_val[:, None] * 1e-7)
    min_val = torch.where(positive, vals, torch.full_like(vals, torch.inf)).amin(dim=-1)
    cond = max_val / min_val.clamp(min=1e-12)
    cond = torch.where(torch.isfinite(cond), cond, torch.full_like(cond, torch.inf))
    return cond


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
        _tau_floor = (
            tau_rfx[..., j : j + 1].to(device=device, dtype=dtype).clamp(min=1e-4, max=10.0) * 0.1
        )
        _search_base = torch.max(sigma_rfx[:, j : j + 1], _tau_floor)
        candidates[:, :, j] = (_search_base * scale_tensor[None]).clamp(min=1e-4, max=20.0)
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
        weights = torch.softmax(
            candidate_target.nan_to_num(nan=-1e30, posinf=1e30, neginf=-1e30), dim=1
        )
        sigma_j_mean = (candidates[:, :, j] * weights).sum(dim=1).clamp(min=1e-4, max=20.0)
        best_target, _ = candidate_target.max(dim=1)
        improve = active_j & torch.isfinite(best_target) & (best_target > target + accept_tol)
        sigma_rfx[:, j] = torch.where(improve, sigma_j_mean, sigma_rfx[:, j])
        target = torch.where(improve, best_target, target)
        changed |= improve

    sigma_rfx = torch.where(active_q, sigma_rfx, sigma_rfx_base)
    return sigma_rfx, target, changed


@torch.no_grad()
def _normalTailBetaCorrection(
    sigma_rfx: torch.Tensor,
    beta_output: torch.Tensor,
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
    prior_capped: torch.Tensor,
    prior_stabilized: torch.Tensor,
    sigma_eps_for_grid: torch.Tensor,
    beta_tail_grid: bool,
    beta_tail_grid_min_d: int,
    beta_tail_grid_min_cond: float,
    beta_tail_grid_blend: float,
    beta_tail_grid_both_trigger_blend: float,
    beta_tail_grid_scales: tuple[float, ...] | list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail β correction: sigma-grid posterior mean, with OLS for double-trigger rows."""
    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    if not beta_tail_grid:
        return beta_output, torch.zeros((B,), device=device, dtype=dtype)
    if mask_d is None:
        active_d = torch.ones_like(beta_output, dtype=torch.bool)
        d_count = torch.full((B,), beta_output.shape[-1], device=device, dtype=torch.long)
    else:
        active_d = mask_d[..., : beta_output.shape[-1]].to(device=device).bool()
        d_count = active_d.sum(dim=-1)
    eligible_d = d_count >= int(beta_tail_grid_min_d)
    if not bool(eligible_d.any()):
        return beta_output, torch.zeros((B,), device=device, dtype=dtype)
    beta_next = beta_output
    gate_out = torch.zeros((B,), device=device, dtype=dtype)
    if bool(eligible_d.any()):
        beta_grid, beta_cond = _normalSigmaGridBetaAverage(
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            sigma_rfx,
            sigma_eps_for_grid,
            nu_ffx,
            tau_ffx,
            family_ffx,
            tau_rfx,
            family_sigma_rfx,
            tau_eps,
            family_sigma_eps,
            mask_d,
            mask_q,
            beta_tail_grid_scales,
            return_condition=True,
        )
        cap_hit = prior_capped.to(device=device) > 0
        stabilized = prior_stabilized.to(device=device) > 0
        high_cond = beta_cond >= float(beta_tail_grid_min_cond)
        gate = eligible_d & (cap_hit | stabilized | high_cond)
        cap_lo = nu_ffx[..., : beta_grid.shape[-1]] - 4.0 * tau_ffx[
            ..., : beta_grid.shape[-1]
        ].clamp(min=1e-4)
        cap_hi = nu_ffx[..., : beta_grid.shape[-1]] + 4.0 * tau_ffx[
            ..., : beta_grid.shape[-1]
        ].clamp(min=1e-4)
        beta_grid = beta_grid.clamp(min=cap_lo, max=cap_hi)
        blend = min(max(float(beta_tail_grid_blend), 0.0), 1.0)
        both_trigger = cap_hit & stabilized
        if bool((both_trigger & eligible_d).any()):
            # Both MAP β-cap AND in-MAP σ-grid stabilization fired → σ_rfx is inflated
            # and GLS collapses toward the prior. The σ-grid GLS is also collapsed in this
            # regime. Use prior-regularized OLS (no σ_rfx dependence) as the blend target.
            d_ols = beta_output.shape[-1]
            mask4_ols = mask_n[:, :, :, None]
            Xm_masked_ols = Xm * mask4_ols
            XtX_ols = torch.einsum('bmnd,bmnk->bdk', Xm_masked_ols, Xm_masked_ols)
            Xty_ols = torch.einsum('bmnd,bmn->bd', Xm_masked_ols, ym)
            se2_ols = sigma_eps_for_grid.to(device=device, dtype=dtype).clamp(min=1e-6).square()
            prior_prec_ols = (
                1.0 / tau_ffx[..., :d_ols].to(device=device, dtype=dtype).clamp(min=1e-4).square()
            )
            A_ols = XtX_ols / se2_ols[:, None, None] + torch.diag_embed(prior_prec_ols)
            b_ols = Xty_ols / se2_ols[:, None] + prior_prec_ols * nu_ffx[..., :d_ols].to(
                device=device, dtype=dtype
            )
            beta_ols = _safeSolve(A_ols + _adaptiveRidge(A_ols), b_ols).nan_to_num(
                nan=0.0, posinf=0.0, neginf=0.0
            )
            beta_ols = beta_ols.clamp(min=cap_lo, max=cap_hi)
            both_blend = min(max(float(beta_tail_grid_both_trigger_blend), 0.0), 1.0)
            beta_grid = torch.where(
                (both_trigger & eligible_d)[:, None] & active_d,
                beta_output + both_blend * (beta_ols - beta_output),
                beta_output + blend * (beta_grid - beta_output),
            )
        else:
            beta_grid = beta_output + blend * (beta_grid - beta_output)
        beta_next = torch.where(gate[:, None] & active_d, beta_grid, beta_output)
        gate_out = gate.to(dtype)
    return beta_next, gate_out


@torch.no_grad()
def _normalSigmaRfxRescue(
    sigma_rfx: torch.Tensor,
    beta_output: torch.Tensor,
    final_target: torch.Tensor,
    active_q: torch.Tensor,
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
    sigma_eps: torch.Tensor,
    log_sigma_eps: torch.Tensor,
    corr: torch.Tensor,
    stuck_ratio: float,
    tau_scales: tuple[float, ...] | list[float],
    accept_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Profile-MAP rescue for σ_rfx components stuck far below the prior scale."""
    B = Xm.shape[0]
    q = sigma_rfx.shape[-1]
    device, dtype = Xm.device, Xm.dtype
    rescue_accept = torch.zeros(B, device=device, dtype=torch.bool)
    tau_q = tau_rfx[..., :q].to(device=device, dtype=dtype).clamp(min=1e-4)
    stuck_j = (sigma_rfx < tau_q * float(stuck_ratio)) & active_q
    stuck_rows = stuck_j.any(dim=-1)
    if not bool(stuck_rows.any()):
        return sigma_rfx, beta_output, final_target, rescue_accept
    _mask4_r = mask_m[:, :, None, None]
    _active_r = mask_m.bool()
    _eye_q_r = torch.eye(q, device=device, dtype=dtype)
    _eye_q_bm_r = _eye_q_r.expand(B, Xm.shape[1], q, q)
    _Xm_s = Xm * _mask4_r
    _Zm_s = Zm * _mask4_r
    _ZtZ_r = torch.einsum('bmnq,bmnr->bmqr', _Zm_s, _Zm_s)
    _ZtZ_safe_r = torch.where(_active_r[:, :, None, None], _ZtZ_r, _eye_q_bm_r)
    _Zty_r = torch.einsum('bmnq,bmn->bmq', _Zm_s, ym)
    _ZtX_r = torch.einsum('bmnq,bmnd->bmqd', _Zm_s, _Xm_s)
    _XtZ_r = _ZtX_r.mT
    _XtX_r = torch.einsum('bmnd,bmnk->bdk', _Xm_s, _Xm_s)
    _Xty_r = torch.einsum('bmnd,bmn->bd', _Xm_s, ym)
    _se2_r = sigma_eps.square()
    _d_r = Xm.shape[-1]
    _pprec = 1.0 / tau_ffx[..., :_d_r].to(device=device, dtype=dtype).clamp(min=1e-4).square()
    _pnu = nu_ffx[..., :_d_r].to(device=device, dtype=dtype)

    def _prof_target(sig_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _sig2 = sig_v.square().clamp(min=1e-8)
        _inner = _ZtZ_safe_r + torch.diag_embed(_se2_r[:, None, None] / _sig2[:, None, :])
        _W_g = _safeSolve(_inner + _adaptiveRidgeBm(_inner), _eye_q_bm_r)
        _W_ZtX = torch.einsum('bmqp,bmpd->bmqd', _W_g, _ZtX_r)
        _corr_XX = torch.einsum('bmdq,bmqk->bdk', _XtZ_r, _W_ZtX)
        _W_Zty = torch.einsum('bmqp,bmp->bmq', _W_g, _Zty_r)
        _corr_Xy = torch.einsum('bmdq,bmq->bd', _XtZ_r, _W_Zty)
        _A = (_XtX_r - _corr_XX) / _se2_r[:, None, None] + torch.diag_embed(_pprec)
        _b = (_Xty_r - _corr_Xy) / _se2_r[:, None] + _pprec * _pnu
        _beta = (
            _safeSolve(_A + _adaptiveRidge(_A), _b)
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(-20.0, 20.0)
        )
        _t = _logMarginalTarget(
            _beta[:, None, :],
            sig_v.clamp(min=1e-4, max=20.0).log()[:, None, :],
            log_sigma_eps[:, None],
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
        return _beta, _t

    _beta_base_r, _t_base_r = _prof_target(sigma_rfx)
    best_sigma_r = sigma_rfx.clone()
    best_beta_r = torch.where(stuck_rows[:, None], _beta_base_r, beta_output)
    best_target_r = torch.where(stuck_rows, _t_base_r, final_target)
    for _s in tau_scales:
        _sig_cand = torch.where(
            stuck_j,
            (tau_q * float(_s)).clamp(min=1e-4, max=20.0),
            sigma_rfx,
        )
        _beta_c, _t_c = _prof_target(_sig_cand)
        _better = stuck_rows & torch.isfinite(_t_c) & (_t_c > best_target_r + accept_tol)
        best_sigma_r = torch.where(_better[:, None], _sig_cand, best_sigma_r)
        best_beta_r = torch.where(_better[:, None], _beta_c, best_beta_r)
        best_target_r = torch.where(_better, _t_c, best_target_r)
    _new_rescue = (best_target_r > final_target + accept_tol) & stuck_rows
    if bool(_new_rescue.any()):
        sigma_rfx = torch.where(_new_rescue[:, None], best_sigma_r, sigma_rfx)
        beta_output = torch.where(_new_rescue[:, None], best_beta_r, beta_output)
        final_target = torch.where(_new_rescue, best_target_r, final_target)
        rescue_accept = rescue_accept | _new_rescue
    return sigma_rfx, beta_output, final_target, rescue_accept


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

    blup_var = gls.blup_var  # σ²·MAP_W_g diagonal; raw blup_var would understate MAP shrinkage

    # delta-method inflation: 1 + 2/(G-d)
    G = mask_m.float().sum(dim=-1)
    df_sigma = (G - float(d)).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma.clamp(min=2.0))[:, None, None]

    # Kackar-Harville correction: β uncertainty propagation into BLUPs
    beta_identified = gls.beta_mask.any(dim=-1)
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

    # n_g-dependent floor; large Ψ/G_mom floor omitted since MAP Ψ is better calibrated
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
    beta_alpha_low: float = 0.65,
    beta_alpha_high: float = 0.75,
    beta_prior_cap: float | None = 4.0,
    beta_sigma_grid: bool = False,
    beta_sigma_grid_scales: tuple[float, ...] | list[float] = (0.75, 1.0, 1.3333333),
    beta_sigma_grid_min_d: int = 5,
) -> dict[str, torch.Tensor]:
    """Refine (β, σ_rfx, σ_eps) via marginal MAP; recompute GLS/BLUP under diagonal MAP Ψ."""
    q = Zm.shape[-1]
    if q == 0 or n_steps <= 0:
        return stats
    corr = _fixedCorrFromStats(stats, eta_rfx, mask_q, q)

    beta = stats['beta_est'].detach().clone().requires_grad_(True)
    log_sigma_rfx = (
        stats['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    log_sigma_eps = (
        stats['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([log_sigma_rfx, beta, log_sigma_eps], lr=lr)
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
    sigma_eps_map = log_sigma_eps.detach().exp()
    beta_det = beta.detach()

    if mask_q is not None:
        sigma_rfx = torch.where(mask_q[..., :q].bool(), sigma_rfx, stats['sigma_rfx_est'][..., :q])
    beta = beta_det
    out = dict(stats)
    # MAP beta overfits the low-d normal sets; keep the older GLS/OLS blend there.
    beta_for_blup_override = beta.detach() if beta.shape[-1] > 4 else None
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
                eligible_d = (d_count >= int(beta_sigma_grid_min_d))[:, None]
                stabilize = cap_components & eligible_d
                beta_grid = _normalSigmaGridBetaAverage(
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    sigma_rfx,
                    sigma_eps_map,
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
    out['normal_map_sigma_rfx'] = sigma_rfx
    out['normal_map_sigma_eps'] = sigma_eps_map.unsqueeze(-1)
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
    moment_blend: float = 1.0,
    prior_weight: float = 4.0,
    recompute_blup: bool = True,
    beta_alpha_low: float = 0.65,
    beta_alpha_high: float = 0.75,
    accept_tol: float = 1e-6,
    sigma_grid_refine: bool = False,
    sigma_grid_scales: tuple[float, ...] | list[float] = (0.75, 1.0, 1.3333333),
    beta_tail_grid: bool = False,
    beta_tail_grid_scales: tuple[float, ...] | list[float] = (0.75, 1.0, 1.3333333),
    beta_tail_grid_min_d: int = 9,
    beta_tail_grid_min_cond: float = 1000.0,
    beta_tail_grid_blend: float = 0.25,
    beta_tail_grid_both_trigger_blend: float = 0.75,
    sigma_grid_max_rounds: int = 1,
    sigma_grid_rescue: bool = False,
    sigma_grid_rescue_stuck_ratio: float = 0.5,
    sigma_grid_rescue_tau_scales: tuple[float, ...] | list[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, torch.Tensor]:
    """Moment-based EB update for diagonal σ_rfx under the exact Gaussian marginal, optionally followed by a coordinate grid pass and tail β correction."""
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
        sigma_candidate = ((1.0 - blend) * sigma_rfx_base.log() + blend * sigma_moment.log()).exp()
        active_q = (
            mask_q[..., :q].to(device=device).bool()
            if mask_q is not None
            else torch.ones(B, q, device=device, dtype=torch.bool)
        )
        final_target = target_for(sigma_candidate, log_sigma_eps_base)
        accept = torch.isfinite(final_target) & (final_target >= base_target - accept_tol)
        sigma_rfx = torch.where(accept[:, None], sigma_candidate, sigma_rfx_base)
        sigma_rfx = torch.where(active_q, sigma_rfx, sigma_rfx_base)
        grid_changed = torch.zeros(B, device=device, dtype=torch.bool)
        if sigma_grid_refine:
            _running_target = torch.where(accept, final_target, base_target)
            _n_rounds = max(1, int(sigma_grid_max_rounds))
            for _round_idx in range(_n_rounds):
                sigma_rfx, _round_best_target, _round_changed = _normalSigmaRfxGridRefine(
                    beta,
                    sigma_rfx,
                    log_sigma_eps_base,
                    _running_target,
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
                grid_changed |= _round_changed
                if not bool(_round_changed.any()):
                    _running_target = _round_best_target
                    break
                # Re-evaluate at the accepted softmax-mean σ_rfx for a correct baseline
                # in the next round.  Without this, the stored target is from the best
                # candidate (not the mean), so subsequent rounds fail to improve.
                if _round_idx + 1 < _n_rounds:
                    _running_target = target_for(sigma_rfx, log_sigma_eps_base)
                else:
                    _running_target = _round_best_target
            final_target = _running_target

        if sigma_grid_rescue:
            sigma_rfx, beta_output, final_target, _rescue_accept = _normalSigmaRfxRescue(
                sigma_rfx,
                beta_output,
                final_target,
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
                tau_eps,
                family_sigma_eps,
                mask_d,
                mask_q,
                sigma_eps=stats['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-6),
                log_sigma_eps=log_sigma_eps_base,
                corr=corr,
                stuck_ratio=sigma_grid_rescue_stuck_ratio,
                tau_scales=sigma_grid_rescue_tau_scales,
                accept_tol=accept_tol,
            )
        else:
            _rescue_accept = torch.zeros(B, device=device, dtype=torch.bool)

    out = dict(stats)
    _prior_capped = stats.get(
        'normal_map_beta_prior_capped', torch.zeros((B,), device=device, dtype=dtype)
    ).to(device=device)
    _prior_stabilized = stats.get(
        'normal_map_beta_stabilized', torch.zeros((B,), device=device, dtype=dtype)
    ).to(device=device)
    _sigma_eps_for_grid = (
        stats.get('normal_map_sigma_eps', stats['sigma_eps_est']).squeeze(-1).detach()
    )
    beta_output_next, beta_tail_gate = _normalTailBetaCorrection(
        sigma_rfx,
        beta_output,
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
        prior_capped=_prior_capped,
        prior_stabilized=_prior_stabilized,
        sigma_eps_for_grid=_sigma_eps_for_grid,
        beta_tail_grid=beta_tail_grid,
        beta_tail_grid_min_d=beta_tail_grid_min_d,
        beta_tail_grid_min_cond=beta_tail_grid_min_cond,
        beta_tail_grid_blend=beta_tail_grid_blend,
        beta_tail_grid_both_trigger_blend=beta_tail_grid_both_trigger_blend,
        beta_tail_grid_scales=beta_tail_grid_scales,
    )
    out['sigma_rfx_est'] = sigma_rfx
    out['beta_est'] = beta_output_next
    out['normal_beta_tail_grid_gate'] = beta_tail_gate
    out['normal_laplace_eb_accept'] = accept.to(dtype)
    out['normal_laplace_eb_sigma_grid_accept'] = grid_changed.to(dtype)
    out['normal_laplace_eb_sigma_rescue'] = _rescue_accept.to(dtype)
    out['normal_laplace_eb_target'] = final_target.detach()
    out['normal_laplace_eb_base_target'] = base_target.detach()
    if 'Psi' in stats:
        if recompute_blup:
            _blup_beta_override = None
            if beta.shape[-1] > 4:
                _blup_beta_override = torch.where(_rescue_accept[:, None], beta_output_next, beta)
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
                beta_override=beta_output_next if beta.shape[-1] > 4 else None,
                beta_for_blup_override=_blup_beta_override,
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
                beta_output_next,
            )
        out['Psi'] = torch.diag_embed(sigma_rfx.square())
    return out
