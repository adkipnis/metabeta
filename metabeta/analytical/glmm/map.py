"""MAP/EB refinements for Bernoulli and Poisson GLMMs."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from metabeta.analytical.constants import (
    _BERNOULLI_PSI_EIG_CAP,
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
    _psdProject,
    _pseudoInverse,
    _safeSolve,
)
from metabeta.utils.families import logProbFfx, logProbSigma

__all__ = [
    'refineBernoulliLaplaceEb',
    'refineBernoulliMapBeta',
    'refineBernoulliNagqSrfx',
    'refineBernoulliNestedBeta',
    'refinePoissonLaplaceEb',
    'refinePoissonMarginalMeanBeta',
    'refinePoissonSigmaGrid',
]


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
    """Refine β via Newton on the exact Bernoulli score at fixed b̂_g, then update b̂_g (Ψ fixed at PQL Laplace M-step). n_outer rounds of alternation are followed by a Ψ M-step with BC1 analytic correction."""
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or n_steps <= 0 or n_outer <= 0:
        return stats

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype

    active_d_mask = normal_prec = is_student = zeros_d = None
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

    blups = stats['blup_est'].detach().clone()
    beta = stats['beta_est'].detach().clone()

    for _outer in range(n_outer):
        # β Newton: n_steps steps at current b̂_g
        eta_rfx = torch.einsum('bmnq,bmq->bmn', Zm, blups)
        for _ in range(n_steps):
            eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + eta_rfx
            mu = torch.sigmoid(eta)
            w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n
            XtWX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
            score = torch.einsum('bmnd,bmn->bd', Xm, (ym - mu) * mask_n)

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

        # b̂_g Newton: n_newton steps at current β (Ψ fixed)
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

    # Ψ M-step with BC1 analytic correction (Breslow-Lin 1995)
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
    mu_f = torch.sigmoid(eta_f)
    w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n
    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q_bm)
    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv_f = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4
    bg_outer_mat = torch.einsum('bmq,bmr->bmqr', blups, blups)
    Psi_lap_raw = (bg_outer_mat + Hg_inv_f).sum(dim=1) / G[:, None, None]

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
    """Refine β via nested Newton: re-converge b̂_g at each β step (INLA-style). Ψ is held fixed at the nAGQ estimate."""
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

        XtWX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        score = torch.einsum('bmnd,bmn->bd', Xm, (ym - mu) * mask_n)

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
        eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, blups)
        mu_f = torch.sigmoid(eta_f)
        w_f = (mu_f * (1.0 - mu_f)).clamp(min=1e-6) * mask_n

    out = dict(stats)
    out['beta_est'] = beta

    if not has_rfx:
        return out

    # M-step + BC1 analytic correction
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
    """Diagonal single-mode Laplace-EB for Bernoulli GLMMs: jointly optimizes β and diagonal σ_rfx under the Laplace-approximated marginal with a σ continuation cap."""
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
) -> torch.Tensor:
    """Approximate log-link marginalization offset 0.5 * diag(Z Ψ Z')."""
    B, _, _, q = Zm.shape
    if mask_q is None:
        active_q = torch.ones(B, q, device=Zm.device, dtype=torch.bool)
    else:
        active_q = mask_q[:, :q].to(device=Zm.device).bool()
    sigma2 = sigma_rfx[:, :q].to(device=Zm.device, dtype=Zm.dtype).square()
    sigma2 = sigma2 * active_q.to(Zm.dtype)
    return 0.5 * torch.einsum('bmnq,bq->bmn', Zm.square(), sigma2)


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
    max_step: float = 1.0,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Gated β update for Poisson marginal means at fixed diagonal Ψ.

    The pseudo target uses E[y|β,Ψ] ≈ exp(Xβ + 0.5 diag(ZΨZ')).  This is not a full
    marginal likelihood; it is a cheap fixed-Ψ correction for rows where conditional
    PQL/EB β is visibly too extreme relative to INLA.
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
    gate = (d_count >= float(min_d)) & active_q.any(dim=1)
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
    offset = _poissonMarginalMeanOffset(stats['sigma_rfx_est'][:, :q].detach(), Zm, mask_q)

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
    """Refine σ_rfx via nAGQ LML gradient. q==1: scalar adaptive GH (k points); q in [2,5]: Cartesian product GH; q>5 unchanged. ZWZ_g is fixed at the PQL mode throughout."""
    q = Zm.shape[-1]
    if q == 0 or 'Psi_lap' not in stats:
        return stats

    B, m = Xm.shape[:2]
    device, dtype = Xm.device, Xm.dtype
    n_max = Zm.shape[2]

    if mask_q is not None:
        active_q_count = mask_q[:, :q].long().sum(dim=-1)
    else:
        active_q_count = torch.full((B,), q, device=device, dtype=torch.long)

    nagq1_eligible = active_q_count == 1
    mv_eligible = (active_q_count >= 2) & (active_q_count <= 5)
    any_eligible = nagq1_eligible | mv_eligible
    if not any_eligible.any():
        return stats

    beta = stats['beta_est'].detach()
    eta_fix = torch.einsum('bmnd,bd->bmn', Xm, beta).detach()
    Psi_lap_new = stats['Psi_lap'].detach().clone()

    # scalar nAGQ (q == 1)
    if nagq1_eligible.any():
        z_np, w_np = np.polynomial.hermite.hermgauss(k)
        z_nodes = torch.tensor(z_np, dtype=dtype, device=device)
        log_w = torch.tensor(np.log(w_np), dtype=dtype, device=device)

        if mask_q is not None:
            q_idx = mask_q[:, :q].long().argmax(dim=-1)
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
                H_g = ZWZ_g1 + 1.0 / s2[:, None].clamp(min=1e-8)
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

    # product GH (q >= 2)
    _K_MV = {2: 5, 3: 5, 4: 3, 5: 3}
    for q_act in range(2, 6):
        elig_bs = (active_q_count == q_act).nonzero(as_tuple=True)[0].tolist()
        if not elig_bs:
            continue
        n_elig = len(elig_bs)

        k_mv = _K_MV[q_act]
        z_grid, log_w_grid = _ghProductGrid([k_mv] * q_act, dtype, device)
        lz2 = z_grid.pow(2).sum(dim=-1)  # ‖z_j‖² correction

        act_idx = []
        for b in elig_bs:
            if mask_q is not None:
                ai = mask_q[b, :q].nonzero(as_tuple=True)[0]
            else:
                ai = torch.arange(q_act, device=device)
            act_idx.append(ai)

        z_cols_e = torch.stack([Zm[b, :, :, act_idx[i]] for i, b in enumerate(elig_bs)])
        b_g0_e = torch.stack(
            [stats['blup_est'][b, :, act_idx[i]] for i, b in enumerate(elig_bs)]
        ).detach()
        eta_fix_e = eta_fix[elig_bs]
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
        ).detach()
        log_s2_e = log_s2_e_init.clone().requires_grad_(True)

        eye_q_act = torch.eye(q_act, device=device, dtype=dtype)
        optimizer_mv = torch.optim.Adam([log_s2_e], lr=lr)

        with torch.enable_grad():
            for _ in range(n_steps):
                optimizer_mv.zero_grad(set_to_none=True)
                s2_e = log_s2_e.exp()

                # H_g = ZWZ_g + diag(1/s2)
                H_g_e = (
                    ZWZ_g_e
                    + torch.diag_embed(1.0 / s2_e.clamp(min=1e-8))[:, None]
                    + 1e-6 * eye_q_act[None, None]
                )
                chol_Hg = torch.linalg.cholesky(H_g_e)
                log_det_Hg = 2.0 * chol_Hg.diagonal(dim1=-2, dim2=-1).log().sum(-1)

                H_g_inv = (
                    torch.cholesky_solve(
                        eye_q_act[None, None].expand(n_elig, m, q_act, q_act), chol_Hg
                    )
                    + 1e-6 * eye_q_act[None, None]
                )
                L_g = torch.linalg.cholesky(H_g_inv)

                b_gj_e = b_g0_e[:, :, :, None] + math.sqrt(2.0) * torch.einsum(
                    'emqr,Kr->emqK', L_g, z_grid
                )
                eta_gj_e = eta_fix_e[:, :, :, None] + torch.einsum(
                    'emnq,emqK->emnK', z_cols_e, b_gj_e
                )

                mn4 = mask_n_e[:, :, :, None]
                ll_gj_e = (ym_e[:, :, :, None] * eta_gj_e - F.softplus(eta_gj_e)) * mn4
                ll_gj_e = ll_gj_e.sum(dim=2)

                lp_gj_e = (
                    -0.5 * q_act * math.log(2.0 * math.pi)
                    - 0.5 * log_s2_e.sum(dim=-1)[:, None, None]
                    - 0.5 * (b_gj_e.pow(2) / s2_e[:, None, :, None].clamp(min=1e-8)).sum(dim=2)
                )

                lml_g_e = (
                    torch.logsumexp(
                        log_w_grid[None, None, :] + ll_gj_e + lp_gj_e + lz2[None, None, :],
                        dim=-1,
                    )
                    + 0.5 * q_act * math.log(2.0)
                    - 0.5 * log_det_Hg
                )

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

    # Newton BLUP refresh under refined Ψ (ineligible rows are restored below)
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
