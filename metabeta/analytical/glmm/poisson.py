"""MAP/EB refinements for Poisson GLMMs."""

import math

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
    'refinePoissonLaplacePirlsDiag',
    'refinePoissonLaplacePirlsFull',
    'refinePoissonLaplacePirlsSigmaGrid',
    'refinePoissonMarginalMeanBeta',
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


def _poissonBetaPriorPrecision(
    beta: torch.Tensor,
    active_d: torch.Tensor,
    nu_ffx: torch.Tensor | None,
    tau_ffx: torch.Tensor | None,
    family_ffx: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local quadratic fixed-effect prior precision and center."""
    B, d = beta.shape
    device, dtype = beta.device, beta.dtype
    zeros_d = torch.zeros(B, d, device=device, dtype=dtype)
    if nu_ffx is None or tau_ffx is None:
        return zeros_d, zeros_d

    nu = nu_ffx[:, :d].to(device=device, dtype=dtype)
    tau = tau_ffx[:, :d].to(device=device, dtype=dtype)
    active_prior = (tau > 0) & active_d
    normal_prec = torch.where(active_prior, 1.0 / tau.clamp(min=1e-4).square(), zeros_d)
    if family_ffx is None:
        return normal_prec, nu

    is_student = (family_ffx == 1).unsqueeze(-1)
    student_prec = torch.where(
        active_prior,
        6.0 / (5.0 * tau.clamp(min=1e-8).square() + (beta - nu).square()).clamp(min=1e-8),
        zeros_d,
    )
    return torch.where(is_student, student_prec, normal_prec), nu


def _poissonJointPirlsStepDiag(
    beta: torch.Tensor,
    blups: torch.Tensor,
    sigma: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    active_d: torch.Tensor,
    active_q: torch.Tensor,
    nu_ffx: torch.Tensor | None,
    tau_ffx: torch.Tensor | None,
    family_ffx: torch.Tensor | None,
    damping: float,
    max_beta_step: float,
    max_blup_step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Schur-complement Newton/PIRLS step for β and group modes."""
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    device, dtype = Xm.device, Xm.dtype
    active_m = mask_m.bool()
    active_q_f = active_q.to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    prec_q = torch.where(active_q, 1.0 / sigma.clamp(min=1e-4).square(), torch.ones_like(sigma))

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    mu, deriv = _poissonMeanDerivative(eta)
    resid_eff = (ym - mu) * deriv * mask_n
    w = (mu * deriv.square()).clamp(min=1e-8) * mask_n

    prior_prec, prior_center = _poissonBetaPriorPrecision(
        beta, active_d, nu_ffx, tau_ffx, family_ffx
    )
    inactive_d_prec = (~active_d).to(dtype)
    inactive_q_prec = (~active_q).to(dtype)

    score_b = torch.einsum('bmnd,bmn->bd', Xm, resid_eff)
    score_b = score_b + prior_prec * (prior_center - beta)
    Hbb = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
    Hbb = Hbb + torch.diag_embed(prior_prec + inactive_d_prec)

    score_u = torch.einsum('bmnq,bmn->bmq', Z_eff, resid_eff)
    score_u = score_u - prec_q[:, None, :] * blups
    score_u = score_u * active_m[:, :, None] * active_q_f[:, None, :]
    A = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    A = A + torch.diag_embed(prec_q + inactive_q_prec)[:, None]
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, m, q, q)
    A = torch.where(active_m[:, :, None, None], A, eye_q)
    A_safe = A + _adaptiveRidgeBm(A)

    Bmat = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w, Z_eff)
    Ainv_score_u = _safeSolve(A_safe, score_u)
    Ainv_BT = _safeSolve(A_safe, Bmat.transpose(-1, -2))

    schur = Hbb - torch.einsum('bmdq,bmqr->bdr', Bmat, Ainv_BT)
    rhs_b = score_b - torch.einsum('bmdq,bmq->bd', Bmat, Ainv_score_u)
    rhs_b = rhs_b * active_d.to(dtype)
    delta_beta = _safeSolve(schur + _adaptiveRidge(schur), rhs_b)
    delta_beta = delta_beta.clamp(-float(max_beta_step), float(max_beta_step))
    delta_beta = delta_beta * active_d.to(dtype)

    Bt_delta = torch.einsum('bmqd,bd->bmq', Bmat.transpose(-1, -2), delta_beta)
    delta_u = Ainv_score_u - _safeSolve(A_safe, Bt_delta)
    delta_u = delta_u.clamp(-float(max_blup_step), float(max_blup_step))
    delta_u = delta_u * active_m[:, :, None] * active_q_f[:, None, :]

    beta_new = beta + float(damping) * delta_beta
    beta_new = beta_new.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta_new = beta_new.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    beta_new = torch.where(active_d, beta_new, beta)
    blups_new = blups + float(damping) * delta_u
    blups_new = blups_new.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups_new = blups_new.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    blups_new = blups_new * active_m[:, :, None] * active_q_f[:, None, :]
    return beta_new, blups_new, A


def _poissonJointHessianDiag(
    beta: torch.Tensor,
    blups: torch.Tensor,
    sigma: torch.Tensor,
    Zm: torch.Tensor,
    Xm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    active_q: torch.Tensor,
) -> torch.Tensor:
    """Return random-effect block Hessians for a diagonal-Σ joint candidate."""
    B, m, _, q = Zm.shape
    device, dtype = Zm.device, Zm.dtype
    active_m = mask_m.bool()
    active_q_f = active_q.to(dtype)
    inactive_q_prec = (~active_q).to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    prec_q = torch.where(active_q, 1.0 / sigma.clamp(min=1e-4).square(), torch.ones_like(sigma))

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    mu, deriv = _poissonMeanDerivative(eta)
    w = (mu * deriv.square()).clamp(min=1e-8) * mask_n
    A = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    A = A + torch.diag_embed(prec_q + inactive_q_prec)[:, None]
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, m, q, q)
    return torch.where(active_m[:, :, None, None], A, eye_q)


def _poissonPreparePsiFull(
    Psi: torch.Tensor,
    active_q: torch.Tensor,
    psi_max: float = _POISSON_PSI_EIG_CAP,
    psi_floor: float = 1e-6,
) -> torch.Tensor:
    """Return a masked positive-definite Ψ with inactive dimensions set to identity."""
    dtype = Psi.dtype
    q = Psi.shape[-1]
    eye_q = torch.eye(q, device=Psi.device, dtype=dtype)
    active_q_f = active_q.to(dtype)
    active_qq = active_q_f[:, :, None] * active_q_f[:, None, :]
    inactive_diag = (~active_q).to(dtype)

    Psi = 0.5 * (Psi + Psi.mT)
    Psi = Psi * active_qq + torch.diag_embed(inactive_diag)
    Psi = _psdClampEigenvalues(Psi, psi_max)
    Psi = Psi * active_qq + torch.diag_embed(inactive_diag)
    Psi = Psi + float(psi_floor) * torch.diag_embed(active_q_f)
    return 0.5 * (Psi + Psi.mT) + 0.0 * eye_q


def _poissonPsiPrecisionFull(
    Psi: torch.Tensor,
    active_q: torch.Tensor,
    psi_max: float = _POISSON_PSI_EIG_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stabilized full Ψ and Ψ⁻¹ for active random-effect dimensions."""
    q = Psi.shape[-1]
    eye_q = torch.eye(q, device=Psi.device, dtype=Psi.dtype).expand(Psi.shape[0], q, q)
    Psi_safe = _poissonPreparePsiFull(Psi, active_q, psi_max=psi_max)
    Psi_inv = _safeSolve(Psi_safe + _adaptiveRidge(Psi_safe), eye_q)
    return Psi_safe, 0.5 * (Psi_inv + Psi_inv.mT)


def _poissonJointPirlsStepFull(
    beta: torch.Tensor,
    blups: torch.Tensor,
    Psi: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    active_d: torch.Tensor,
    active_q: torch.Tensor,
    nu_ffx: torch.Tensor | None,
    tau_ffx: torch.Tensor | None,
    family_ffx: torch.Tensor | None,
    damping: float,
    max_beta_step: float,
    max_blup_step: float,
    psi_max: float = _POISSON_PSI_EIG_CAP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Schur-complement Newton/PIRLS step for β and modes under full Ψ."""
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    device, dtype = Xm.device, Xm.dtype
    active_m = mask_m.bool()
    active_q_f = active_q.to(dtype)
    active_qq = active_q_f[:, :, None] * active_q_f[:, None, :]
    Z_eff = Zm * active_q_f[:, None, None, :]
    Psi_safe, Psi_inv = _poissonPsiPrecisionFull(Psi, active_q, psi_max=psi_max)

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    mu, deriv = _poissonMeanDerivative(eta)
    resid_eff = (ym - mu) * deriv * mask_n
    w = (mu * deriv.square()).clamp(min=1e-8) * mask_n

    prior_prec, prior_center = _poissonBetaPriorPrecision(
        beta, active_d, nu_ffx, tau_ffx, family_ffx
    )
    inactive_d_prec = (~active_d).to(dtype)

    score_b = torch.einsum('bmnd,bmn->bd', Xm, resid_eff)
    score_b = score_b + prior_prec * (prior_center - beta)
    Hbb = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
    Hbb = Hbb + torch.diag_embed(prior_prec + inactive_d_prec)

    score_u = torch.einsum('bmnq,bmn->bmq', Z_eff, resid_eff)
    score_u = score_u - torch.einsum('bqr,bmr->bmq', Psi_inv, blups)
    score_u = score_u * active_m[:, :, None] * active_q_f[:, None, :]
    A = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    A = A + Psi_inv[:, None]
    A = A * active_qq[:, None] + torch.diag_embed((~active_q).to(dtype))[:, None]
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, m, q, q)
    A = torch.where(active_m[:, :, None, None], A, eye_q)
    A_safe = A + _adaptiveRidgeBm(A)

    Bmat = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w, Z_eff)
    Ainv_score_u = _safeSolve(A_safe, score_u)
    Ainv_BT = _safeSolve(A_safe, Bmat.transpose(-1, -2))

    schur = Hbb - torch.einsum('bmdq,bmqr->bdr', Bmat, Ainv_BT)
    rhs_b = score_b - torch.einsum('bmdq,bmq->bd', Bmat, Ainv_score_u)
    rhs_b = rhs_b * active_d.to(dtype)
    delta_beta = _safeSolve(schur + _adaptiveRidge(schur), rhs_b)
    delta_beta = delta_beta.clamp(-float(max_beta_step), float(max_beta_step))
    delta_beta = delta_beta * active_d.to(dtype)

    Bt_delta = torch.einsum('bmqd,bd->bmq', Bmat.transpose(-1, -2), delta_beta)
    delta_u = Ainv_score_u - _safeSolve(A_safe, Bt_delta)
    delta_u = delta_u.clamp(-float(max_blup_step), float(max_blup_step))
    delta_u = delta_u * active_m[:, :, None] * active_q_f[:, None, :]

    beta_new = beta + float(damping) * delta_beta
    beta_new = beta_new.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta_new = beta_new.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    beta_new = torch.where(active_d, beta_new, beta)
    blups_new = blups + float(damping) * delta_u
    blups_new = blups_new.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups_new = blups_new.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    blups_new = blups_new * active_m[:, :, None] * active_q_f[:, None, :]
    return beta_new, blups_new, A, Psi_safe


def _poissonJointHessianFull(
    beta: torch.Tensor,
    blups: torch.Tensor,
    Psi: torch.Tensor,
    Zm: torch.Tensor,
    Xm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    active_q: torch.Tensor,
    psi_max: float = _POISSON_PSI_EIG_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random-effect block Hessians for a full-Ψ joint candidate."""
    B, m, _, q = Zm.shape
    device, dtype = Zm.device, Zm.dtype
    active_m = mask_m.bool()
    active_q_f = active_q.to(dtype)
    active_qq = active_q_f[:, :, None] * active_q_f[:, None, :]
    Z_eff = Zm * active_q_f[:, None, None, :]
    Psi_safe, Psi_inv = _poissonPsiPrecisionFull(Psi, active_q, psi_max=psi_max)

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    mu, deriv = _poissonMeanDerivative(eta)
    w = (mu * deriv.square()).clamp(min=1e-8) * mask_n
    A = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    A = A + Psi_inv[:, None]
    A = A * active_qq[:, None] + torch.diag_embed((~active_q).to(dtype))[:, None]
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, m, q, q)
    A = torch.where(active_m[:, :, None, None], A, eye_q)
    return A, Psi_safe


def _poissonLaplaceEbTargetFull(
    beta: torch.Tensor,
    Psi_lap: torch.Tensor,
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
    psi_max: float = _POISSON_PSI_EIG_CAP,
) -> torch.Tensor:
    """Laplace-approximated log posterior target for full-Ψ Poisson GLMMs."""
    B, _, _, d = Xm.shape
    q = Zm.shape[-1]
    dtype = Xm.dtype
    active_q_f = active_q.to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    Psi_safe, Psi_inv = _poissonPsiPrecisionFull(Psi_lap, active_q, psi_max=psi_max)

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Z_eff, blups)
    eta_eff = eta.clamp(max=_POISSON_ETA_CLIP_MAX)
    ll_g = ((ym * eta_eff - torch.exp(eta_eff)) * mask_n).sum(dim=-1)

    q_count = active_q_f.sum(dim=-1)
    sign_psi, log_det_psi = torch.linalg.slogdet(Psi_safe)
    log_det_psi = torch.where(sign_psi > 0, log_det_psi, log_det_psi.new_zeros(()))
    quad_u = torch.einsum('bmq,bqr,bmr->bm', blups, Psi_inv, blups)
    log_prior_b = -0.5 * (
        q_count[:, None] * math.log(2.0 * math.pi) + log_det_psi[:, None] + quad_u
    )

    sign_h, log_det_H = torch.linalg.slogdet(H)
    log_det_H = torch.where(sign_h > 0, log_det_H, log_det_H.new_zeros(()))
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

    sigma_diag = Psi_safe.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()
    if tau_rfx is not None and family_sigma_rfx is not None:
        target = target + logProbSigma(
            sigma_diag.unsqueeze(1),
            tau_rfx[:, :q].clamp(min=1e-8).unsqueeze(1),
            family_sigma_rfx,
            active_q_f.unsqueeze(1),
        ).squeeze(1)
    if sigma_log_jacobian:
        target = target + (sigma_diag.clamp(min=1e-4).log() * active_q_f).sum(dim=-1)

    return target


def refinePoissonLaplacePirlsDiag(
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
    n_outer: int = 4,
    n_pirls: int = 1,
    n_final: int = 2,
    damping: float = 0.5,
    sigma_blend: float = 0.5,
    sigma_prior_weight: float = 4.0,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Fixed-budget diagonal-Σ Laplace-PIRLS with EB covariance updates.

    This is the first Poisson joint-geometry prototype: β, group modes, and diagonal σ are
    alternated inside one candidate, then the full candidate is accepted/rejected by the
    existing diagonal Laplace target.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_outer <= 0:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    active_d = (
        mask_d[:, :d].to(device=device).bool()
        if mask_d is not None
        else torch.ones(B, d, device=device, dtype=torch.bool)
    )
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )
    active_q_f = active_q.to(dtype)
    active_m = mask_m.bool()
    m_active = mask_m.to(dtype).sum(dim=1, keepdim=True).clamp(min=1.0)

    beta_base = stats['beta_est'][:, :d].detach()
    beta = beta_base.clone().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta = beta.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    if 'blup_est' in stats:
        blups = stats['blup_est'][:, :, :q].detach().clone()
    else:
        blups = Zm.new_zeros(B, Zm.shape[1], q)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups = blups.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    blups = blups * mask_m[:, :, None] * active_q_f[:, None, :]
    if 'sigma_rfx_est' in stats:
        sigma_base = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)
    else:
        sigma_base = Zm.new_full((B, q), 0.1).clamp(min=1e-4, max=sigma_max)
    sigma = sigma_base.clone()

    sigma0 = sigma_base
    if tau_rfx is not None:
        sigma0 = tau_rfx[:, :q].to(device=device, dtype=dtype).clamp(min=1e-4, max=sigma_max)

    A = None
    for _ in range(n_outer):
        for _ in range(max(int(n_pirls), 1)):
            beta, blups, A = _poissonJointPirlsStepDiag(
                beta,
                blups,
                sigma,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                active_d,
                active_q,
                nu_ffx,
                tau_ffx,
                family_ffx,
                damping=damping,
                max_beta_step=max_beta_step,
                max_blup_step=max_blup_step,
            )
        eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
        A_inv = _safeSolve(A + _adaptiveRidgeBm(A), eye_q) * mask_m[:, :, None, None]
        second_moment = blups.square() + A_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
        second_moment = second_moment * mask_m[:, :, None] * active_q_f[:, None, :]
        sigma2_moment = (second_moment.sum(dim=1) + float(sigma_prior_weight) * sigma0.square()) / (
            m_active + float(sigma_prior_weight)
        )
        sigma_moment = sigma2_moment.clamp(min=1e-8, max=sigma_max**2).sqrt()
        log_sigma = (1.0 - float(sigma_blend)) * sigma.clamp(min=1e-4).log()
        log_sigma = log_sigma + float(sigma_blend) * sigma_moment.clamp(min=1e-4).log()
        sigma = log_sigma.exp().clamp(min=1e-4, max=sigma_max)
        sigma = torch.where(active_q, sigma, sigma_base)

    for _ in range(max(int(n_final), 0)):
        beta, blups, A = _poissonJointPirlsStepDiag(
            beta,
            blups,
            sigma,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            active_d,
            active_q,
            nu_ffx,
            tau_ffx,
            family_ffx,
            damping=damping,
            max_beta_step=max_beta_step,
            max_blup_step=max_blup_step,
        )

    if A is None:
        _, A, _ = _poissonLaplaceModeDiag(
            beta,
            sigma.clamp(min=1e-4, max=sigma_max).log(),
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            mask_q,
            n_inner=1,
            damping=damping,
        )

    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
    H_inv = _safeSolve(A + _adaptiveRidgeBm(A), eye_q) * mask_m[:, :, None, None]
    blup_var = H_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q_f[:, None, :]
    Psi_lap = torch.diag_embed(sigma.square())

    final_target = _poissonLaplaceEbTargetDiag(
        beta,
        sigma.clamp(min=1e-4, max=sigma_max).log(),
        blups,
        A,
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
        sigma_log_jacobian=True,
    )
    base_target = torch.full((B,), float('nan'), device=device, dtype=dtype)
    accept = torch.ones(B, device=device, dtype=torch.bool)
    if accept_only_improved and 'sigma_rfx_est' in stats:
        base_log_sigma = sigma_base.clamp(min=1e-4, max=sigma_max).log()
        base_blups, base_H, base_active_q = _poissonLaplaceModeDiag(
            beta_base,
            base_log_sigma,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            mask_q,
            n_inner=max(int(n_final), 1),
            damping=damping,
        )
        base_target = _poissonLaplaceEbTargetDiag(
            beta_base,
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
            sigma_log_jacobian=True,
        )
        accept = final_target >= base_target - 1e-5

    out = dict(stats)
    out['beta_est'] = torch.where(accept[:, None], beta, beta_base)
    out['sigma_rfx_est'] = torch.where(accept[:, None], sigma, sigma_base)
    out['blup_est'] = torch.where(accept[:, None, None], blups, stats['blup_est'][:, :, :q])
    if 'blup_var' in stats:
        out['blup_var'] = torch.where(accept[:, None, None], blup_var, stats['blup_var'][:, :, :q])
    else:
        out['blup_var'] = blup_var
    out['Psi_lap'] = torch.where(accept[:, None, None], Psi_lap, stats['Psi_lap'][:, :q, :q])
    out['Psi_lap'] = _psdClampEigenvalues(out['Psi_lap'], _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_laplace_pirls_accept'] = accept.to(dtype)
        out['poisson_laplace_pirls_target'] = final_target
        out['poisson_laplace_pirls_base_target'] = base_target
    return out


def refinePoissonLaplacePirlsFull(
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
    n_outer: int = 4,
    n_pirls: int = 1,
    n_final: int = 2,
    damping: float = 0.5,
    psi_blend: float = 0.5,
    psi_prior_weight: float = 4.0,
    psi_max: float = _POISSON_PSI_EIG_CAP,
    offdiag_shrink: float = 0.0,
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Fixed-budget full-Σ Laplace-PIRLS with EB covariance updates.

    This variant keeps the same joint β/u PIRLS geometry as the diagonal path, but estimates
    the full random-effect covariance from posterior second moments inside the outer loop.
    It is intentionally a separate candidate path until its accuracy/stability is known.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_outer <= 0:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    active_d = (
        mask_d[:, :d].to(device=device).bool()
        if mask_d is not None
        else torch.ones(B, d, device=device, dtype=torch.bool)
    )
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )
    active_q_f = active_q.to(dtype)
    active_qq = active_q_f[:, :, None] * active_q_f[:, None, :]
    active_m = mask_m.bool()
    m_active = mask_m.to(dtype).sum(dim=1, keepdim=True).clamp(min=1.0)

    beta_base = stats['beta_est'][:, :d].detach()
    beta = beta_base.clone().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    beta = beta.clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    if 'blup_est' in stats:
        blups = stats['blup_est'][:, :, :q].detach().clone()
    else:
        blups = Zm.new_zeros(B, Zm.shape[1], q)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups = blups.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    blups = blups * mask_m[:, :, None] * active_q_f[:, None, :]

    if 'Psi_lap' in stats:
        Psi_base = stats['Psi_lap'][:, :q, :q].detach()
    elif 'sigma_rfx_est' in stats:
        Psi_base = torch.diag_embed(stats['sigma_rfx_est'][:, :q].detach().square())
    else:
        Psi_base = torch.eye(q, device=device, dtype=dtype).expand(B, q, q) * 0.01
    Psi_base = _poissonPreparePsiFull(Psi_base, active_q, psi_max=psi_max)
    Psi = Psi_base.clone()

    if tau_rfx is not None:
        sigma0 = (
            tau_rfx[:, :q].to(device=device, dtype=dtype).clamp(min=1e-4, max=math.sqrt(psi_max))
        )
        Psi0 = torch.diag_embed(sigma0.square())
    else:
        Psi0 = Psi_base
    Psi0 = _poissonPreparePsiFull(Psi0, active_q, psi_max=psi_max)

    A = None
    for _ in range(n_outer):
        for _ in range(max(int(n_pirls), 1)):
            beta, blups, A, Psi = _poissonJointPirlsStepFull(
                beta,
                blups,
                Psi,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                active_d,
                active_q,
                nu_ffx,
                tau_ffx,
                family_ffx,
                damping=damping,
                max_beta_step=max_beta_step,
                max_blup_step=max_blup_step,
                psi_max=psi_max,
            )
        eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
        A_inv = _safeSolve(A + _adaptiveRidgeBm(A), eye_q) * mask_m[:, :, None, None]
        second_moment = torch.einsum('bmq,bmr->bmqr', blups, blups) + A_inv
        second_moment = second_moment * mask_m[:, :, None, None] * active_qq[:, None]
        Psi_moment = (second_moment.sum(dim=1) + float(psi_prior_weight) * Psi0) / (
            m_active[:, :, None] + float(psi_prior_weight)
        )
        if offdiag_shrink > 0.0:
            diag_moment = torch.diag_embed(Psi_moment.diagonal(dim1=-2, dim2=-1))
            Psi_moment = (1.0 - float(offdiag_shrink)) * Psi_moment
            Psi_moment = Psi_moment + float(offdiag_shrink) * diag_moment
        Psi = (1.0 - float(psi_blend)) * Psi + float(psi_blend) * Psi_moment
        Psi = _poissonPreparePsiFull(Psi, active_q, psi_max=psi_max)

    for _ in range(max(int(n_final), 0)):
        beta, blups, A, Psi = _poissonJointPirlsStepFull(
            beta,
            blups,
            Psi,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            active_d,
            active_q,
            nu_ffx,
            tau_ffx,
            family_ffx,
            damping=damping,
            max_beta_step=max_beta_step,
            max_blup_step=max_blup_step,
            psi_max=psi_max,
        )

    if A is None:
        A, Psi = _poissonJointHessianFull(
            beta, blups, Psi, Zm, Xm, mask_n, mask_m, active_q, psi_max=psi_max
        )

    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
    H_inv = _safeSolve(A + _adaptiveRidgeBm(A), eye_q) * mask_m[:, :, None, None]
    blup_var = H_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q_f[:, None, :]
    sigma = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()
    sigma_base = Psi_base.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()

    final_target = _poissonLaplaceEbTargetFull(
        beta,
        Psi,
        blups,
        A,
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
        sigma_log_jacobian=True,
        psi_max=psi_max,
    )
    base_A, base_Psi = _poissonJointHessianFull(
        beta_base,
        stats.get('blup_est', blups)[:, :, :q]
        .detach()
        .clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
        * mask_m[:, :, None]
        * active_q_f[:, None, :],
        Psi_base,
        Zm,
        Xm,
        mask_n,
        mask_m,
        active_q,
        psi_max=psi_max,
    )
    base_target = _poissonLaplaceEbTargetFull(
        beta_base,
        base_Psi,
        stats.get('blup_est', blups)[:, :, :q]
        .detach()
        .clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
        * mask_m[:, :, None]
        * active_q_f[:, None, :],
        base_A,
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
        sigma_log_jacobian=True,
        psi_max=psi_max,
    )
    accept = torch.isfinite(final_target)
    if accept_only_improved:
        accept = accept & (final_target >= base_target - 1e-5)

    out = dict(stats)
    out['beta_est'] = torch.where(accept[:, None], beta, beta_base)
    out['sigma_rfx_est'] = torch.where(accept[:, None], sigma, sigma_base)
    out['blup_est'] = torch.where(accept[:, None, None], blups, stats['blup_est'][:, :, :q])
    if 'blup_var' in stats:
        out['blup_var'] = torch.where(accept[:, None, None], blup_var, stats['blup_var'][:, :, :q])
    else:
        out['blup_var'] = blup_var
    out['Psi_lap'] = torch.where(accept[:, None, None], Psi, Psi_base)
    out['Psi_lap'] = _poissonPreparePsiFull(out['Psi_lap'], active_q, psi_max=psi_max)
    if return_diagnostics:
        out['poisson_laplace_pirls_full_accept'] = accept.to(dtype)
        out['poisson_laplace_pirls_full_target'] = final_target
        out['poisson_laplace_pirls_full_base_target'] = base_target
    return out


def refinePoissonLaplacePirlsSigmaGrid(
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
    scales: tuple[float, ...] = (0.5, 0.75, 1.0),
    n_steps: int = 2,
    damping: float = 0.5,
    min_d: int = 1,
    max_q: int | None = None,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Full-candidate sigma grid for the diagonal Laplace-PIRLS Poisson path.

    This pass scores and writes back the whole candidate: β, diagonal σ, BLUPs, and BLUP
    variances. Each σ scale is followed by a few joint PIRLS β/u steps at fixed σ, then
    accepted by the same diagonal Laplace target used by PIRLS. The default grid is
    conservative: it allows shrinkage or fixed-σ re-synchronization, but avoids σ-inflation
    candidates that created rare large σ outliers in diagnostics.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_steps <= 0 or 'sigma_rfx_est' not in stats:
        return stats

    B = Xm.shape[0]
    device, dtype = Xm.device, Xm.dtype
    active_d = (
        mask_d[:, :d].to(device=device).bool()
        if mask_d is not None
        else torch.ones(B, d, device=device, dtype=torch.bool)
    )
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )
    d_count = active_d.long().sum(dim=1)
    q_count = active_q.long().sum(dim=1)
    gate = (d_count >= int(min_d)) & (q_count >= 1)
    if max_q is not None:
        gate = gate & (q_count <= int(max_q))
    if not gate.any():
        out = dict(stats)
        if return_diagnostics:
            out['poisson_pirls_sigma_grid_gate'] = gate.to(dtype)
            out['poisson_pirls_sigma_grid_accept'] = torch.zeros(B, device=device, dtype=dtype)
            out['poisson_pirls_sigma_grid_scale'] = torch.ones(B, device=device, dtype=dtype)
        return out

    beta_base = stats['beta_est'][:, :d].detach().clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    sigma_base = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)
    blups_base = stats.get('blup_est', Zm.new_zeros(B, Zm.shape[1], q))[:, :, :q].detach()
    blups_base = blups_base.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups_base = blups_base.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    blups_base = blups_base * mask_m[:, :, None] * active_q.to(dtype)[:, None, :]

    base_A = _poissonJointHessianDiag(
        beta_base, blups_base, sigma_base, Zm, Xm, mask_n, mask_m, active_q
    )
    base_target = _poissonLaplaceEbTargetDiag(
        beta_base,
        sigma_base.log(),
        blups_base,
        base_A,
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
        sigma_log_jacobian=True,
    )

    best_beta = beta_base.clone()
    best_sigma = sigma_base.clone()
    best_blups = blups_base.clone()
    best_A = base_A.clone()
    best_target = base_target.clone()
    best_scale = torch.ones(B, device=device, dtype=dtype)

    for scale in scales:
        scale_f = float(scale)
        sigma_try = torch.where(
            gate[:, None] & active_q,
            (sigma_base * scale_f).clamp(min=1e-4, max=sigma_max),
            sigma_base,
        )
        beta_try = beta_base.clone()
        blups_try = blups_base.clone()
        A_try = base_A
        for _ in range(max(int(n_steps), 1)):
            beta_try, blups_try, A_try = _poissonJointPirlsStepDiag(
                beta_try,
                blups_try,
                sigma_try,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                active_d,
                active_q,
                nu_ffx,
                tau_ffx,
                family_ffx,
                damping=damping,
                max_beta_step=max_beta_step,
                max_blup_step=max_blup_step,
            )
        target_try = _poissonLaplaceEbTargetDiag(
            beta_try,
            sigma_try.log(),
            blups_try,
            A_try,
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
            sigma_log_jacobian=True,
        )
        improve = gate & torch.isfinite(target_try) & (target_try > best_target + 1e-5)
        best_target = torch.where(improve, target_try, best_target)
        best_beta = torch.where(improve[:, None], beta_try, best_beta)
        best_sigma = torch.where(improve[:, None], sigma_try, best_sigma)
        best_blups = torch.where(improve[:, None, None], blups_try, best_blups)
        best_A = torch.where(improve[:, None, None, None], A_try, best_A)
        best_scale = torch.where(improve, torch.full_like(best_scale, scale_f), best_scale)

    accept = gate & (best_scale != 1.0)
    if accept_only_improved:
        accept = gate & (best_target > base_target + 1e-5)

    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
    H_inv = _safeSolve(best_A + _adaptiveRidgeBm(best_A), eye_q) * mask_m[:, :, None, None]
    blup_var = H_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q.to(dtype)[:, None, :]
    Psi_lap = torch.diag_embed(best_sigma.square())

    out = dict(stats)
    out['beta_est'] = torch.where(accept[:, None], best_beta, beta_base)
    out['sigma_rfx_est'] = torch.where(accept[:, None], best_sigma, sigma_base)
    out['blup_est'] = torch.where(accept[:, None, None], best_blups, blups_base)
    if 'blup_var' in stats:
        out['blup_var'] = torch.where(accept[:, None, None], blup_var, stats['blup_var'][:, :, :q])
    else:
        out['blup_var'] = blup_var
    out['Psi_lap'] = torch.where(accept[:, None, None], Psi_lap, stats['Psi_lap'][:, :q, :q])
    out['Psi_lap'] = _psdClampEigenvalues(out['Psi_lap'], _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_pirls_sigma_grid_gate'] = gate.to(dtype)
        out['poisson_pirls_sigma_grid_accept'] = accept.to(dtype)
        out['poisson_pirls_sigma_grid_scale'] = best_scale
        out['poisson_pirls_sigma_grid_target'] = best_target
        out['poisson_pirls_sigma_grid_base_target'] = base_target
    return out


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
