"""MAP/EB refinements for Poisson GLMMs."""

import math

import numpy as np
import torch

from metabeta.analytical.constants import (
    _POISSON_BETA_CLAMP,
    _POISSON_BLUP_CLAMP,
    _POISSON_ETA_CLIP_MAX,
    _POISSON_PSI_EIG_CAP,
)
from metabeta.analytical.glmm.irls import _poissonMeanDerivative
from metabeta.analytical.linalg import (
    _adaptiveRidge,
    _adaptiveRidgeBm,
    _psdClampEigenvalues,
    _safeSolve,
)
from metabeta.utils.families import logProbFfx, logProbSigma

__all__ = [
    'refinePoissonLaplaceEb',
    'refinePoissonAgqBeta',
    'refinePoissonMarginalMeanBeta',
    'refinePoissonSigmaGrid',
]


def _poissonLaplaceModeDiag(
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
    """Approximate b_g MAP modes and Hessians for diagonal-Ψ Poisson GLMMs."""
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
        mu, deriv = _poissonMeanDerivative(eta)
        score_g = torch.einsum('bmnq,bmn->bmq', Z_eff, (ym - mu) * deriv * mask_n)
        score_g = score_g - prec[:, None, :] * blups
        w = (mu * deriv.square()).clamp(min=1e-8) * mask_n
        ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
        H = ZWZ + torch.diag_embed(prec)[:, None]
        H_safe = torch.where(active[:, :, None, None], H, eye_q_bm)
        delta = _safeSolve(H_safe + _adaptiveRidgeBm(H_safe), score_g)
        blups = (blups + damping * delta) * mask_m[:, :, None] * active_q[:, None, :].to(dtype)
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        blups = blups.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    mu, deriv = _poissonMeanDerivative(eta)
    w = (mu * deriv.square()).clamp(min=1e-8) * mask_n
    ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    H = ZWZ + torch.diag_embed(prec)[:, None]
    H = torch.where(active[:, :, None, None], H, eye_q_bm)
    return blups, H, active_q


def _poissonLaplaceEbTargetDiag(
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
    """Laplace-approximated log posterior target for diagonal Poisson GLMMs."""
    B, _, _, d = Xm.shape
    q = Zm.shape[-1]
    dtype = Xm.dtype
    active_q_f = active_q.to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    eta_eff = eta.clamp(max=_POISSON_ETA_CLIP_MAX)
    ll = (ym * eta_eff - torch.exp(eta_eff)) * mask_n
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


def _poissonMarginalMeanOffset(
    sigma_rfx: torch.Tensor,
    Zm: torch.Tensor,
    mask_q: torch.Tensor | None,
    Psi_lap: torch.Tensor | None = None,
    full_psi_min_q: int = 3,
) -> torch.Tensor:
    """Approximate log-link marginalization offset 0.5 * diag(Z Ψ Z')."""
    B, _, _, q = Zm.shape
    if mask_q is None:
        active_q = torch.ones(B, q, device=Zm.device, dtype=torch.bool)
    else:
        active_q = mask_q[:, :q].to(device=Zm.device).bool()
    sigma2 = sigma_rfx[:, :q].to(device=Zm.device, dtype=Zm.dtype).square()
    sigma2 = sigma2 * active_q.to(Zm.dtype)
    offset_diag = 0.5 * torch.einsum('bmnq,bq->bmn', Zm.square(), sigma2)

    if Psi_lap is None:
        return offset_diag

    q_count = active_q.long().sum(dim=1)
    full_gate = q_count >= int(full_psi_min_q)
    if not full_gate.any():
        return offset_diag

    Psi = Psi_lap[:, :q, :q].detach().to(device=Zm.device, dtype=Zm.dtype)
    active_q_f = active_q.to(Zm.dtype)
    Psi = Psi * active_q_f[:, :, None] * active_q_f[:, None, :]
    Psi = _psdClampEigenvalues(
        Psi.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), _POISSON_PSI_EIG_CAP
    )
    offset_full = 0.5 * torch.einsum('bmnq,bqr,bmnr->bmn', Zm, Psi, Zm).clamp(min=0.0)
    return torch.where(full_gate[:, None, None], offset_full, offset_diag)


def _poissonMarginalMeanBetaTarget(
    beta: torch.Tensor,
    offset: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask_n: torch.Tensor,
    nu_ffx: torch.Tensor | None,
    tau_ffx: torch.Tensor | None,
    family_ffx: torch.Tensor | None,
    mask_d: torch.Tensor | None,
) -> torch.Tensor:
    """Pseudo marginal Poisson log posterior for β at fixed diagonal Ψ."""
    B, _, _, d = Xm.shape
    dtype = Xm.dtype
    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + offset
    eta_eff = eta.clamp(max=_POISSON_ETA_CLIP_MAX)
    target = ((ym * eta_eff - torch.exp(eta_eff)) * mask_n).sum(dim=(1, 2))

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
    return target


def refinePoissonMarginalMeanBeta(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor | None = None,
    tau_ffx: torch.Tensor | None = None,
    family_ffx: torch.Tensor | None = None,
    mask_d: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    n_steps: int = 4,
    damping: float = 0.7,
    min_d: int = 5,
    max_q: int | None = None,
    max_step: float = 1.0,
    accept_only_improved: bool = True,
    marginal_psi_lap: torch.Tensor | None = None,
    full_psi_min_q: int = 3,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Gated β update for Poisson marginal means at fixed Ψ.

    The pseudo target uses E[y|β,Ψ] ≈ exp(Xβ + 0.5 diag(ZΨZ')).  By default it uses the
    diagonal EB Ψ; passing ``marginal_psi_lap`` lets high-q rows use the full PQL covariance
    in the offset without writing covariance, σ, or BLUP changes back to the output.  This is
    not a full marginal likelihood; it is a cheap fixed-Ψ correction for rows where
    conditional PQL/EB β is visibly too extreme relative to INLA.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_steps <= 0 or 'sigma_rfx_est' not in stats:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    if mask_d is None:
        active_d = torch.ones(B, d, device=device, dtype=torch.bool)
    else:
        active_d = mask_d[:, :d].to(device=device).bool()
    if mask_q is None:
        active_q = torch.ones(B, q, device=device, dtype=torch.bool)
    else:
        active_q = mask_q[:, :q].to(device=device).bool()
    d_count = active_d.to(dtype).sum(dim=1)
    q_count = active_q.long().sum(dim=1)
    gate = (d_count >= float(min_d)) & (q_count >= 1)
    if max_q is not None:
        gate = gate & (q_count <= int(max_q))
    if not gate.any():
        out = dict(stats)
        if return_diagnostics:
            out['poisson_marginal_beta_gate'] = gate.to(dtype)
            out['poisson_marginal_beta_accept'] = torch.zeros(B, device=device, dtype=dtype)
            out['poisson_marginal_beta_jump'] = torch.full(
                (B,), float('nan'), device=device, dtype=dtype
            )
        return out

    base_beta = stats['beta_est'][:, :d].detach()
    beta = base_beta.clone().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta = beta.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    offset = _poissonMarginalMeanOffset(
        stats['sigma_rfx_est'][:, :q].detach(),
        Zm,
        mask_q,
        Psi_lap=marginal_psi_lap,
        full_psi_min_q=full_psi_min_q,
    )

    zeros_d = Xm.new_zeros(B, d)
    normal_prec = prior_prec = is_student = None
    if nu_ffx is not None and tau_ffx is not None:
        tau = tau_ffx[:, :d].to(device=device, dtype=dtype)
        active_prior = (tau > 0) & active_d
        normal_prec = torch.where(active_prior, 1.0 / tau.clamp(min=1e-4).square(), zeros_d)
        is_student = (family_ffx == 1).unsqueeze(-1) if family_ffx is not None else None

    inactive_prec = (~active_d).to(dtype)
    for _ in range(n_steps):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + offset
        mu, deriv = _poissonMeanDerivative(eta)
        score = torch.einsum('bmnd,bmn->bd', Xm, (ym - mu) * deriv * mask_n)
        w = (mu * deriv.square()).clamp(min=1e-8) * mask_n
        XtWX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)

        if normal_prec is not None:
            if is_student is not None:
                tau = tau_ffx[:, :d].to(device=device, dtype=dtype)
                nu = nu_ffx[:, :d].to(device=device, dtype=dtype)
                active_prior = (tau > 0) & active_d
                student_prec = torch.where(
                    active_prior,
                    6.0
                    / (5.0 * tau.clamp(min=1e-8).square() + (beta - nu).square()).clamp(min=1e-8),
                    zeros_d,
                )
                prior_prec = torch.where(is_student, student_prec, normal_prec)
            else:
                nu = nu_ffx[:, :d].to(device=device, dtype=dtype)
                prior_prec = normal_prec
            XtWX = XtWX + torch.diag_embed(prior_prec)
            score = score + prior_prec * (nu - beta)

        XtWX = XtWX + torch.diag_embed(inactive_prec)
        score = score * active_d.to(dtype) * gate[:, None].to(dtype)
        delta = _safeSolve(XtWX + _adaptiveRidge(XtWX), score)
        delta = delta.clamp(-float(max_step), float(max_step))
        beta = beta + float(damping) * delta
        beta = beta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        beta = beta.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
        beta = torch.where(active_d & gate[:, None], beta, base_beta)

    final_target = _poissonMarginalMeanBetaTarget(
        beta, offset, Xm, ym, mask_n, nu_ffx, tau_ffx, family_ffx, mask_d
    )
    base_target = _poissonMarginalMeanBetaTarget(
        base_beta, offset, Xm, ym, mask_n, nu_ffx, tau_ffx, family_ffx, mask_d
    )
    accept = gate
    if accept_only_improved:
        accept = accept & (final_target >= base_target - 1e-5)
    beta_final = torch.where(accept[:, None], beta, base_beta)

    out = dict(stats)
    out['beta_est'] = beta_final
    if return_diagnostics:
        jump = ((beta_final - base_beta).square() * active_d.to(dtype)).sum(dim=1)
        jump = (jump / d_count.clamp(min=1.0)).sqrt()
        out['poisson_marginal_beta_gate'] = gate.to(dtype)
        out['poisson_marginal_beta_accept'] = accept.to(dtype)
        out['poisson_marginal_beta_jump'] = jump
        out['poisson_marginal_beta_target'] = final_target
        out['poisson_marginal_beta_base_target'] = base_target
    return out


def _poissonAgqTargetDiag(
    beta: torch.Tensor,
    sigma_rfx: torch.Tensor,
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
    mask_q: torch.Tensor | None,
    k: int,
    n_inner: int,
) -> torch.Tensor:
    """Small-q adaptive GH marginal log posterior for candidate β and diagonal σ."""
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    device, dtype = Xm.device, Xm.dtype
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )
    q_count = active_q.long().sum(dim=1)
    target = torch.full((B,), -float('inf'), device=device, dtype=dtype)

    for q_act in range(1, min(q, 2) + 1):
        rows = (q_count == q_act).nonzero(as_tuple=True)[0].tolist()
        if not rows:
            continue
        z_grid, log_w_grid = _ghProductGrid([k] * q_act, dtype, device)
        lz2 = z_grid.square().sum(dim=-1)
        for b in rows:
            idx = active_q[b].nonzero(as_tuple=True)[0]
            beta_b = beta[b]
            sigma_b = sigma_rfx[b, idx].clamp(min=1e-4, max=math.sqrt(_POISSON_PSI_EIG_CAP))
            log_sigma_b = sigma_b.log().view(1, q_act)
            Xm_b = Xm[b : b + 1, :, :, :d]
            ym_b = ym[b : b + 1]
            Zm_b = Zm[b : b + 1, :, :, idx]
            mask_n_b = mask_n[b : b + 1]
            mask_m_b = mask_m[b : b + 1]
            blups_b, H_b, _ = _poissonLaplaceModeDiag(
                beta_b.view(1, d),
                log_sigma_b,
                Xm_b,
                ym_b,
                Zm_b,
                mask_n_b,
                mask_m_b,
                None,
                n_inner=n_inner,
                damping=0.5,
            )
            eye = torch.eye(q_act, device=device, dtype=dtype).expand(1, m, q_act, q_act)
            H_inv = _safeSolve(H_b + _adaptiveRidgeBm(H_b), eye)
            L = torch.linalg.cholesky(_psdClampEigenvalues(H_inv, _POISSON_PSI_EIG_CAP))
            b_nodes = blups_b[:, :, :, None] + math.sqrt(2.0) * torch.einsum(
                'bmqr,Kr->bmqK', L, z_grid
            )
            eta = torch.einsum('bmnd,d->bmn', Xm_b, beta_b)
            eta_nodes = eta[:, :, :, None] + torch.einsum('bmnq,bmqK->bmnK', Zm_b, b_nodes)
            eta_eff = eta_nodes.clamp(max=_POISSON_ETA_CLIP_MAX)
            ll = (ym_b[:, :, :, None] * eta_eff - torch.exp(eta_eff)) * mask_n_b[:, :, :, None]
            ll_g = ll.sum(dim=2)
            lp_g = (
                -0.5 * q_act * math.log(2.0 * math.pi)
                - sigma_b.log().sum()
                - 0.5 * (b_nodes.square() / sigma_b.view(1, 1, q_act, 1).square()).sum(dim=2)
            )
            sign, log_det_H = torch.linalg.slogdet(H_b)
            log_det_H = torch.where(sign > 0, log_det_H, log_det_H.new_zeros(()))
            lml_g = (
                torch.logsumexp(log_w_grid.view(1, 1, -1) + ll_g + lp_g + lz2.view(1, 1, -1), -1)
                + 0.5 * q_act * math.log(2.0)
                - 0.5 * log_det_H
            )
            target_b = (lml_g * mask_m_b).sum()
            if nu_ffx is not None and tau_ffx is not None and family_ffx is not None:
                if mask_d is None:
                    mask_d_b = torch.ones(1, 1, d, device=device, dtype=dtype)
                else:
                    mask_d_b = mask_d[b : b + 1, :d].to(device=device, dtype=dtype).unsqueeze(1)
                target_b = (
                    target_b
                    + logProbFfx(
                        beta_b.view(1, 1, d),
                        nu_ffx[b : b + 1, :d].unsqueeze(1),
                        tau_ffx[b : b + 1, :d].clamp(min=1e-8).unsqueeze(1),
                        family_ffx[b : b + 1],
                        mask_d_b,
                    ).squeeze()
                )
            if tau_rfx is not None and family_sigma_rfx is not None:
                target_b = (
                    target_b
                    + logProbSigma(
                        sigma_b.view(1, 1, q_act),
                        tau_rfx[b : b + 1, idx].clamp(min=1e-8).unsqueeze(1),
                        family_sigma_rfx[b : b + 1],
                        torch.ones(1, 1, q_act, device=device, dtype=dtype),
                    ).squeeze()
                )
            target[b] = target_b
    return target


def refinePoissonAgqBeta(
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
    min_d: int = 1,
    max_q: int = 2,
    n_steps: int = 8,
    lr: float = 0.03,
    max_step: float = 0.75,
    agq_k: int = 5,
    agq_inner: int = 6,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """β-only refinement against the small-q adaptive GH marginal target."""
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_steps <= 0 or 'sigma_rfx_est' not in stats:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    if mask_d is None:
        active_d = torch.ones(B, d, device=device, dtype=torch.bool)
    else:
        active_d = mask_d[:, :d].to(device=device).bool()
    if mask_q is None:
        active_q = torch.ones(B, q, device=device, dtype=torch.bool)
    else:
        active_q = mask_q[:, :q].to(device=device).bool()
    d_count = active_d.long().sum(dim=1)
    q_count = active_q.long().sum(dim=1)
    gate = (d_count >= int(min_d)) & (q_count >= 1) & (q_count <= int(max_q))
    if not gate.any():
        out = dict(stats)
        if return_diagnostics:
            out['poisson_agq_beta_gate'] = gate.to(dtype)
            out['poisson_agq_beta_accept'] = torch.zeros(B, device=device, dtype=dtype)
        return out

    base_beta = stats['beta_est'][:, :d].detach().clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    sigma = (
        stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=math.sqrt(_POISSON_PSI_EIG_CAP))
    )
    base_target = _poissonAgqTargetDiag(
        base_beta,
        sigma,
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
        mask_q,
        k=agq_k,
        n_inner=agq_inner,
    )

    beta = base_beta.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=float(lr))
    min_beta = base_beta - float(max_step)
    max_beta = base_beta + float(max_step)

    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            target = _poissonAgqTargetDiag(
                beta,
                sigma,
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
                mask_q,
                k=agq_k,
                n_inner=agq_inner,
            )
            loss = -target[gate].sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([beta], max_norm=10.0)
            optimizer.step()
            with torch.no_grad():
                beta.copy_(
                    beta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    .clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
                    .clamp(min=min_beta, max=max_beta)
                )
                beta.copy_(torch.where((active_d & gate[:, None]), beta, base_beta))

    beta_final = beta.detach()
    final_target = _poissonAgqTargetDiag(
        beta_final,
        sigma,
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
        mask_q,
        k=agq_k,
        n_inner=agq_inner,
    )
    accept = gate
    if accept_only_improved:
        accept = accept & (final_target >= base_target - 1e-5)
    out = dict(stats)
    out['beta_est'] = torch.where(accept[:, None], beta_final, base_beta)
    if return_diagnostics:
        out['poisson_agq_beta_gate'] = gate.to(dtype)
        out['poisson_agq_beta_accept'] = accept.to(dtype)
        out['poisson_agq_beta_target'] = final_target
        out['poisson_agq_beta_base_target'] = base_target
    return out


def refinePoissonSigmaGrid(
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
    scales: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0, 1.3333333),
    min_d: int = 5,
    max_q: int = 2,
    beta_steps: int = 4,
    beta_damping: float = 0.7,
    beta_max_step: float = 1.0,
    agq_k: int = 3,
    agq_inner: int = 6,
    update_sigma: bool = False,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Targeted σ-offset grid with AGQ acceptance for Poisson rows where q is tiny."""
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or 'sigma_rfx_est' not in stats:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    if mask_d is None:
        active_d = torch.ones(B, d, device=device, dtype=torch.bool)
    else:
        active_d = mask_d[:, :d].to(device=device).bool()
    if mask_q is None:
        active_q = torch.ones(B, q, device=device, dtype=torch.bool)
    else:
        active_q = mask_q[:, :q].to(device=device).bool()
    d_count = active_d.long().sum(dim=1)
    q_count = active_q.long().sum(dim=1)
    gate = (d_count >= int(min_d)) & (q_count >= 1) & (q_count <= int(max_q))
    if not gate.any():
        out = dict(stats)
        if return_diagnostics:
            out['poisson_sigma_grid_gate'] = gate.to(dtype)
            out['poisson_sigma_grid_accept'] = torch.zeros(B, device=device, dtype=dtype)
            out['poisson_sigma_grid_scale'] = torch.ones(B, device=device, dtype=dtype)
        return out

    sigma_base = (
        stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=math.sqrt(_POISSON_PSI_EIG_CAP))
    )
    beta_base = stats['beta_est'][:, :d].detach()
    best_beta = beta_base.clone()
    best_sigma = sigma_base.clone()
    best_target = _poissonAgqTargetDiag(
        beta_base,
        sigma_base,
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
        mask_q,
        k=agq_k,
        n_inner=agq_inner,
    )
    best_scale = torch.ones(B, device=device, dtype=dtype)

    for scale in scales:
        scale_f = float(scale)
        if abs(scale_f - 1.0) < 1e-8:
            continue
        sigma_try = sigma_base.clone()
        sigma_try = torch.where(
            gate[:, None] & active_q,
            (sigma_base * scale_f).clamp(min=1e-4, max=math.sqrt(_POISSON_PSI_EIG_CAP)),
            sigma_try,
        )
        candidate_stats = dict(stats)
        candidate_stats['sigma_rfx_est'] = sigma_try
        candidate_stats = refinePoissonMarginalMeanBeta(
            candidate_stats,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            nu_ffx=nu_ffx,
            tau_ffx=tau_ffx,
            family_ffx=family_ffx,
            mask_d=mask_d,
            mask_q=mask_q,
            n_steps=beta_steps,
            damping=beta_damping,
            min_d=min_d,
            max_step=beta_max_step,
            accept_only_improved=False,
            return_diagnostics=False,
        )
        beta_try = candidate_stats['beta_est'][:, :d].detach()
        target_try = _poissonAgqTargetDiag(
            beta_try,
            sigma_try,
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
            mask_q,
            k=agq_k,
            n_inner=agq_inner,
        )
        improve = gate & (target_try > best_target + 1e-5)
        best_target = torch.where(improve, target_try, best_target)
        best_beta = torch.where(improve[:, None], beta_try, best_beta)
        best_sigma = torch.where(improve[:, None], sigma_try, best_sigma)
        best_scale = torch.where(improve, torch.full_like(best_scale, scale_f), best_scale)

    accept = gate & (best_scale != 1.0)
    out = dict(stats)
    out['beta_est'] = torch.where(accept[:, None], best_beta, beta_base)
    if update_sigma:
        out['sigma_rfx_est'] = torch.where(accept[:, None], best_sigma, sigma_base)
        out['Psi_lap'] = torch.diag_embed(out['sigma_rfx_est'].square())
    if return_diagnostics:
        out['poisson_sigma_grid_gate'] = gate.to(dtype)
        out['poisson_sigma_grid_accept'] = accept.to(dtype)
        out['poisson_sigma_grid_scale'] = best_scale
        out['poisson_sigma_grid_target'] = best_target
    return out


def refinePoissonLaplaceEb(
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
    lr: float = 0.03,
    damping: float = 0.5,
    sigma_start: float = 0.03,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    sigma_log_jacobian: bool = True,
    accept_only_improved: bool = True,
    blup_fallback_beta_jump: float | None = None,
    sigma_prior_cap: float | None = None,
    sigma_prior_cap_min_d: int | None = None,
    recompute_blup_after_calibration: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Diagonal single-mode Laplace-EB for Poisson GLMMs."""
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

    beta = stats['beta_est'][:, :d].detach().clone()
    beta = beta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta = beta.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    beta.requires_grad_(True)

    sigma0 = Zm.new_full((B, q), float(sigma_start)).clamp(min=1e-4, max=sigma_max)
    if 'sigma_rfx_est' in stats:
        sigma0 = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)
    log_sigma = sigma0.log().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([beta, log_sigma], lr=lr)
    min_log_sigma = math.log(1e-4)
    final_log_cap = math.log(sigma_max)
    best_loss = float('inf')
    stale_steps = 0
    n_steps_run = 0

    with torch.enable_grad():
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            log_sigma_step = log_sigma.clamp(min=min_log_sigma, max=final_log_cap)
            blups, H, active_q_step = _poissonLaplaceModeDiag(
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
            target = _poissonLaplaceEbTargetDiag(
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
            if loss_value < best_loss - 1e-4:
                best_loss = loss_value
                stale_steps = 0
            else:
                stale_steps += 1
            with torch.no_grad():
                beta.clamp_(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
                log_sigma.clamp_(min_log_sigma, final_log_cap)
                log_sigma.masked_fill_(~active_q, min_log_sigma)
            if step >= max(n_steps // 2, 1) and stale_steps >= 3:
                break

    with torch.no_grad():
        beta_final = beta.detach()
        log_sigma_final = log_sigma.detach().clamp(min=min_log_sigma, max=final_log_cap)
        blups, H, _ = _poissonLaplaceModeDiag(
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
        sigma_prior_capped = torch.zeros(B, device=device, dtype=torch.bool)
        beta_jump = torch.full((B,), float('nan'), device=device, dtype=dtype)
        final_target = torch.full((B,), float('nan'), device=device, dtype=dtype)
        base_target = torch.full((B,), float('nan'), device=device, dtype=dtype)
        base_beta = stats['beta_est'][:, :d].detach() if 'beta_est' in stats else None
        if accept_only_improved and 'sigma_rfx_est' in stats:
            base_log_sigma = (
                stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max).log()
            )
            base_blups, base_H, base_active_q = _poissonLaplaceModeDiag(
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
            final_target = _poissonLaplaceEbTargetDiag(
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
            base_target = _poissonLaplaceEbTargetDiag(
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

        if sigma_prior_cap is not None and tau_rfx is not None:
            sigma_cap = float(sigma_prior_cap) * tau_rfx[:, :q].to(device=device).clamp(min=1e-4)
            sigma_capped = torch.minimum(sigma, sigma_cap)
            cap_active = active_q & accept[:, None] & (sigma_capped < sigma)
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
                blups_new, H_new, _ = _poissonLaplaceModeDiag(
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
                if blup_fallback.any() and 'blup_var' in stats:
                    blups = torch.where(
                        blup_fallback[:, None, None], stats['blup_est'][:, :, :q], blups
                    )
                    blup_var = torch.where(
                        blup_fallback[:, None, None], stats['blup_var'][:, :, :q], blup_var
                    )

    out = dict(stats)
    out['beta_est'] = beta_final
    out['sigma_rfx_est'] = sigma
    out['blup_est'] = blups.detach()
    out['blup_var'] = blup_var.detach()
    out['Psi_lap'] = _psdClampEigenvalues(Psi_lap, _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['laplace_eb_accept'] = accept.to(dtype)
        out['laplace_eb_steps'] = torch.full((B,), float(n_steps_run), device=device, dtype=dtype)
        out['laplace_eb_target'] = final_target
        out['laplace_eb_base_target'] = base_target
        out['laplace_eb_blup_fallback'] = blup_fallback.to(dtype)
        out['laplace_eb_beta_jump'] = beta_jump
        out['laplace_eb_sigma_prior_capped'] = sigma_prior_capped.to(dtype)
    return out


def _ghProductGrid(
    k_vals: list[int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian-product Gauss-Hermite grid; returns z_nodes (K, q) and log_w (K,)."""
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
