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
    'popPoissonRefinementOptions',
    'refinePoissonLaplaceEb',
    'refinePoissonLaplacePirlsDiag',
    'refinePoissonLaplacePirlsSigmaAverage',
    'refinePoissonLaplacePirlsSigmaGrid',
    'refinePoissonMarginalMeanBeta',
    'refinePoissonPath',
    'refinePoissonVariationalGaussian',
    'refinePoissonVariationalGaussianPolish',
    'refinePoissonVariationalGaussianSigmaAverage',
    'refinePoissonVariationalGaussianStateAverage',
]

_POISSON_LAPLACE_EB_DEFAULTS = {
    'poisson_laplace_eb_steps': 24,
    'poisson_laplace_eb_inner': 4,
    'poisson_laplace_eb_final': 6,
    'poisson_laplace_eb_lr': 0.05,
    'poisson_laplace_eb_blup_fallback_beta_jump': 0.0,
    'poisson_laplace_eb_sigma_prior_cap': 2.5,
    'poisson_laplace_eb_sigma_prior_cap_min_d': 5,
}


def _poissonLaplaceEbMode(value: bool | str) -> str:
    if isinstance(value, bool):
        return 'all' if value else 'off'
    if isinstance(value, str):
        value = value.lower()
        if value in {'poisson_eb', 'calibrated', 'cal', 'default'}:
            return 'cal'
        if value in {'all', 'true', 'yes', 'on'}:
            return 'all'
        if value in {'off', 'false', 'no'}:
            return 'off'
    raise ValueError("poisson_laplace_eb must be bool, 'poisson_eb', 'calibrated', or 'off'")


def _poissonRefinementKwarg(
    kwargs: dict,
    key: str,
    default,
    preset: dict[str, int | float],
):
    return kwargs.pop(key, preset.get(key, default))


def popPoissonRefinementOptions(kwargs: dict, likelihood_family: int) -> dict:
    """Pop Poisson refinement options from ``kwargs`` for the public GLMM dispatcher."""
    poisson_laplace_eb_default = 'poisson_eb' if likelihood_family == 2 else False
    poisson_laplace_eb = kwargs.pop('poisson_laplace_eb', poisson_laplace_eb_default)
    poisson_laplace_eb_mode = _poissonLaplaceEbMode(poisson_laplace_eb)
    poisson_laplace_eb_preset = (
        _POISSON_LAPLACE_EB_DEFAULTS if poisson_laplace_eb_mode == 'cal' else {}
    )
    if poisson_laplace_eb_mode == 'cal':
        poisson_laplace_eb_mode = 'all'

    poisson_laplace_pirls_diag = kwargs.pop('poisson_laplace_pirls_diag', likelihood_family == 2)
    poisson_laplace_pirls_sigma_average_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_scales',
        (0.5, 0.75, 1.0, 1.3333333, 2.0),
    )
    poisson_laplace_pirls_sigma_average_scale_mode = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_scale_mode',
        'scalar',
    )
    poisson_laplace_pirls_sigma_average_intercept_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_intercept_scales',
        (0.75, 1.0, 1.3333333),
    )
    poisson_laplace_pirls_sigma_average_slope_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_slope_scales',
        (0.5, 1.0, 1.5),
    )

    return {
        'poisson_laplace_eb_mode': poisson_laplace_eb_mode,
        'poisson_laplace_eb_diagnostics': kwargs.pop('poisson_laplace_eb_diagnostics', False),
        'poisson_laplace_eb_steps': _poissonRefinementKwarg(
            kwargs, 'poisson_laplace_eb_steps', 12, poisson_laplace_eb_preset
        ),
        'poisson_laplace_eb_inner': _poissonRefinementKwarg(
            kwargs, 'poisson_laplace_eb_inner', 4, poisson_laplace_eb_preset
        ),
        'poisson_laplace_eb_final': _poissonRefinementKwarg(
            kwargs, 'poisson_laplace_eb_final', 6, poisson_laplace_eb_preset
        ),
        'poisson_laplace_eb_lr': _poissonRefinementKwarg(
            kwargs, 'poisson_laplace_eb_lr', 0.03, poisson_laplace_eb_preset
        ),
        'poisson_laplace_eb_blup_fallback_beta_jump': _poissonRefinementKwarg(
            kwargs,
            'poisson_laplace_eb_blup_fallback_beta_jump',
            None,
            poisson_laplace_eb_preset,
        ),
        'poisson_laplace_eb_sigma_prior_cap': _poissonRefinementKwarg(
            kwargs, 'poisson_laplace_eb_sigma_prior_cap', None, poisson_laplace_eb_preset
        ),
        'poisson_laplace_eb_sigma_prior_cap_min_d': _poissonRefinementKwarg(
            kwargs,
            'poisson_laplace_eb_sigma_prior_cap_min_d',
            None,
            poisson_laplace_eb_preset,
        ),
        'poisson_laplace_pirls_diag': poisson_laplace_pirls_diag,
        'poisson_laplace_pirls_diag_outer': kwargs.pop('poisson_laplace_pirls_diag_outer', 4),
        'poisson_laplace_pirls_diag_inner': kwargs.pop('poisson_laplace_pirls_diag_inner', 1),
        'poisson_laplace_pirls_diag_final': kwargs.pop('poisson_laplace_pirls_diag_final', 2),
        'poisson_laplace_pirls_diag_damping': kwargs.pop(
            'poisson_laplace_pirls_diag_damping',
            0.5,
        ),
        'poisson_laplace_pirls_diag_sigma_blend': kwargs.pop(
            'poisson_laplace_pirls_diag_sigma_blend',
            0.5,
        ),
        'poisson_laplace_pirls_diag_prior_weight': kwargs.pop(
            'poisson_laplace_pirls_diag_prior_weight',
            4.0,
        ),
        'poisson_laplace_pirls_sigma_grid': kwargs.pop(
            'poisson_laplace_pirls_sigma_grid',
            bool(poisson_laplace_pirls_diag),
        ),
        'poisson_laplace_pirls_sigma_grid_scales': kwargs.pop(
            'poisson_laplace_pirls_sigma_grid_scales',
            (0.5, 0.75, 1.0),
        ),
        'poisson_laplace_pirls_sigma_grid_steps': kwargs.pop(
            'poisson_laplace_pirls_sigma_grid_steps',
            2,
        ),
        'poisson_laplace_pirls_sigma_grid_min_d': kwargs.pop(
            'poisson_laplace_pirls_sigma_grid_min_d',
            1,
        ),
        'poisson_laplace_pirls_sigma_grid_max_q': kwargs.pop(
            'poisson_laplace_pirls_sigma_grid_max_q',
            None,
        ),
        'poisson_laplace_pirls_sigma_average': kwargs.pop(
            'poisson_laplace_pirls_sigma_average',
            bool(poisson_laplace_pirls_diag),
        ),
        'poisson_laplace_pirls_sigma_average_scales': (poisson_laplace_pirls_sigma_average_scales),
        'poisson_laplace_pirls_sigma_average_scale_mode': (
            poisson_laplace_pirls_sigma_average_scale_mode
        ),
        'poisson_laplace_pirls_sigma_average_intercept_scales': (
            poisson_laplace_pirls_sigma_average_intercept_scales
        ),
        'poisson_laplace_pirls_sigma_average_slope_scales': (
            poisson_laplace_pirls_sigma_average_slope_scales
        ),
        'poisson_laplace_pirls_sigma_average_steps': kwargs.pop(
            'poisson_laplace_pirls_sigma_average_steps',
            2,
        ),
        'poisson_laplace_pirls_sigma_average_temperature': kwargs.pop(
            'poisson_laplace_pirls_sigma_average_temperature',
            2.0,
        ),
        'poisson_laplace_pirls_sigma_average_min_d': kwargs.pop(
            'poisson_laplace_pirls_sigma_average_min_d',
            1,
        ),
        'poisson_laplace_pirls_sigma_average_max_q': kwargs.pop(
            'poisson_laplace_pirls_sigma_average_max_q',
            None,
        ),
        'poisson_laplace_pirls_sigma_average_output_mode': kwargs.pop(
            'poisson_laplace_pirls_sigma_average_output_mode',
            'beta_sigma',
        ),
        'poisson_variational_gaussian': kwargs.pop(
            'poisson_variational_gaussian',
            likelihood_family == 2,
        ),
        'poisson_variational_gaussian_outer': kwargs.pop('poisson_variational_gaussian_outer', 5),
        'poisson_variational_gaussian_inner': kwargs.pop('poisson_variational_gaussian_inner', 5),
        'poisson_variational_gaussian_final': kwargs.pop('poisson_variational_gaussian_final', 2),
        'poisson_variational_gaussian_damping': kwargs.pop(
            'poisson_variational_gaussian_damping',
            0.7,
        ),
        'poisson_variational_gaussian_sigma_blend': kwargs.pop(
            'poisson_variational_gaussian_sigma_blend',
            0.25,
        ),
        'poisson_variational_gaussian_prior_weight': kwargs.pop(
            'poisson_variational_gaussian_prior_weight',
            4.0,
        ),
        'poisson_variational_gaussian_adaptive_steps': kwargs.pop(
            'poisson_variational_gaussian_adaptive_steps',
            2,
        ),
        'poisson_variational_gaussian_adaptive_target_gain': kwargs.pop(
            'poisson_variational_gaussian_adaptive_target_gain',
            1e-4,
        ),
        'poisson_variational_gaussian_adaptive_min_beta_step': kwargs.pop(
            'poisson_variational_gaussian_adaptive_min_beta_step',
            0.01,
        ),
        'poisson_variational_gaussian_adaptive_min_mean_step': kwargs.pop(
            'poisson_variational_gaussian_adaptive_min_mean_step',
            0.01,
        ),
        'poisson_variational_gaussian_adaptive_min_offset_step': kwargs.pop(
            'poisson_variational_gaussian_adaptive_min_offset_step',
            0.01,
        ),
        'poisson_variational_gaussian_adaptive_min_sigma_step': kwargs.pop(
            'poisson_variational_gaussian_adaptive_min_sigma_step',
            0.005,
        ),
        'poisson_variational_gaussian_offset_clip': kwargs.pop(
            'poisson_variational_gaussian_offset_clip',
            None,
        ),
        'poisson_variational_gaussian_sigma_average': kwargs.pop(
            'poisson_variational_gaussian_sigma_average',
            likelihood_family == 2,
        ),
        'poisson_variational_gaussian_sigma_average_scales': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_scales',
            poisson_laplace_pirls_sigma_average_scales,
        ),
        'poisson_variational_gaussian_sigma_average_scale_mode': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_scale_mode',
            poisson_laplace_pirls_sigma_average_scale_mode,
        ),
        'poisson_variational_gaussian_sigma_average_intercept_scales': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_intercept_scales',
            poisson_laplace_pirls_sigma_average_intercept_scales,
        ),
        'poisson_variational_gaussian_sigma_average_slope_scales': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_slope_scales',
            poisson_laplace_pirls_sigma_average_slope_scales,
        ),
        'poisson_variational_gaussian_sigma_average_steps': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_steps',
            2,
        ),
        'poisson_variational_gaussian_sigma_average_temperature': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_temperature',
            2.0,
        ),
        'poisson_variational_gaussian_sigma_average_output_mode': kwargs.pop(
            'poisson_variational_gaussian_sigma_average_output_mode',
            'beta',
        ),
        'poisson_variational_gaussian_state_average': kwargs.pop(
            'poisson_variational_gaussian_state_average',
            False,
        ),
        'poisson_variational_gaussian_state_average_v_scales': kwargs.pop(
            'poisson_variational_gaussian_state_average_v_scales',
            (0.75, 1.0, 1.3333333),
        ),
        'poisson_variational_gaussian_state_average_steps': kwargs.pop(
            'poisson_variational_gaussian_state_average_steps',
            2,
        ),
        'poisson_variational_gaussian_state_average_temperature': kwargs.pop(
            'poisson_variational_gaussian_state_average_temperature',
            2.0,
        ),
        'poisson_variational_gaussian_state_average_output_mode': kwargs.pop(
            'poisson_variational_gaussian_state_average_output_mode',
            'beta',
        ),
        'poisson_variational_gaussian_polish': kwargs.pop(
            'poisson_variational_gaussian_polish',
            False,
        ),
        'poisson_variational_gaussian_polish_steps': kwargs.pop(
            'poisson_variational_gaussian_polish_steps',
            2,
        ),
        'poisson_variational_gaussian_polish_dampings': kwargs.pop(
            'poisson_variational_gaussian_polish_dampings',
            (0.25, 0.5, 0.75, 1.0),
        ),
        'poisson_variational_gaussian_polish_target_gain': kwargs.pop(
            'poisson_variational_gaussian_polish_target_gain',
            1e-4,
        ),
        'poisson_marginal_beta': kwargs.pop('poisson_marginal_beta', likelihood_family == 2),
        'poisson_marginal_beta_steps': kwargs.pop('poisson_marginal_beta_steps', 4),
        'poisson_marginal_beta_damping': kwargs.pop('poisson_marginal_beta_damping', 0.7),
        'poisson_marginal_beta_min_d': kwargs.pop('poisson_marginal_beta_min_d', 1),
        'poisson_marginal_beta_max_q': kwargs.pop('poisson_marginal_beta_max_q', None),
        'poisson_marginal_beta_max_step': kwargs.pop('poisson_marginal_beta_max_step', 1.0),
    }


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


def _poissonVariationalOffset(
    V_g: torch.Tensor,
    Zm: torch.Tensor,
    active_q: torch.Tensor,
    max_offset: float | None = None,
) -> torch.Tensor:
    """Return 0.5 * diag(Z V_g Z') for q(u_g)=N(m_g,V_g)."""
    active_q_f = active_q.to(Zm.dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    offset = 0.5 * torch.einsum('bmnq,bmqr,bmnr->bmn', Z_eff, V_g, Z_eff).clamp(min=0.0)
    if max_offset is not None:
        offset = offset.clamp(max=float(max_offset))
    return offset


def _poissonVariationalHessianDiag(
    beta: torch.Tensor,
    means: torch.Tensor,
    V_g: torch.Tensor,
    sigma: torch.Tensor,
    Zm: torch.Tensor,
    Xm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    active_q: torch.Tensor,
    offset_clip: float | None = None,
) -> torch.Tensor:
    """Return local random-effect precision under the Gaussian variational mean."""
    B, m, _, q = Zm.shape
    device, dtype = Zm.device, Zm.dtype
    active_m = mask_m.bool()
    active_q_f = active_q.to(dtype)
    inactive_q_prec = (~active_q).to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    prec_q = torch.where(active_q, 1.0 / sigma.clamp(min=1e-4).square(), torch.ones_like(sigma))

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
    eta = eta + torch.einsum('bmnq,bmq->bmn', Z_eff, means)
    eta = eta + _poissonVariationalOffset(V_g, Zm, active_q, offset_clip)
    mu, deriv = _poissonMeanDerivative(eta)
    w = (mu * deriv.square()).clamp(min=1e-8) * mask_n
    A = torch.einsum('bmnq,bmn,bmnr->bmqr', Z_eff, w, Z_eff)
    A = A + torch.diag_embed(prec_q + inactive_q_prec)[:, None]
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, m, q, q)
    return torch.where(active_m[:, :, None, None], A, eye_q)


def _poissonVariationalCovarianceDiag(
    beta: torch.Tensor,
    means: torch.Tensor,
    V_g: torch.Tensor,
    sigma: torch.Tensor,
    Zm: torch.Tensor,
    Xm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    active_q: torch.Tensor,
    offset_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refresh q(u) covariance at the current variational state."""
    B, m, _, q = Zm.shape
    device, dtype = Zm.device, Zm.dtype
    active_q_f = active_q.to(dtype)
    A = _poissonVariationalHessianDiag(
        beta, means, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
    )
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, m, q, q)
    V_new = _safeSolve(A + _adaptiveRidgeBm(A), eye_q)
    V_new = V_new * mask_m[:, :, None, None]
    V_new = V_new * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]
    return A, V_new


def _poissonVariationalSigmaMomentDiag(
    means: torch.Tensor,
    V_g: torch.Tensor,
    sigma: torch.Tensor,
    sigma0: torch.Tensor,
    mask_m: torch.Tensor,
    active_q: torch.Tensor,
    sigma_blend: float,
    sigma_prior_weight: float,
    sigma_max: float,
) -> torch.Tensor:
    """Moment/prior update for diagonal random-effect σ in log space."""
    dtype = means.dtype
    active_q_f = active_q.to(dtype)
    m_active = mask_m.to(dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    second_moment = means.square() + V_g.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    second_moment = second_moment * mask_m[:, :, None] * active_q_f[:, None, :]
    sigma2_moment = (second_moment.sum(dim=1) + float(sigma_prior_weight) * sigma0.square()) / (
        m_active + float(sigma_prior_weight)
    )
    sigma_moment = sigma2_moment.clamp(min=1e-8, max=sigma_max**2).sqrt()
    log_sigma = (1.0 - float(sigma_blend)) * sigma.clamp(min=1e-4).log()
    log_sigma = log_sigma + float(sigma_blend) * sigma_moment.clamp(min=1e-4).log()
    sigma_new = log_sigma.exp().clamp(min=1e-4, max=sigma_max)
    return torch.where(active_q, sigma_new, sigma)


def _rowRmsMasked(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return per-row RMS under a broadcast-compatible mask."""
    mask_f = mask.to(delta.dtype)
    while mask_f.ndim < delta.ndim:
        mask_f = mask_f.unsqueeze(-1)
    mask_f = mask_f.expand_as(delta)
    axes = tuple(range(1, delta.ndim))
    denom = mask_f.sum(dim=axes).clamp(min=1.0)
    numer = delta.square().mul(mask_f).sum(dim=axes)
    return (numer / denom).clamp(min=0.0).sqrt()


def _poissonVariationalStepDiag(
    beta: torch.Tensor,
    means: torch.Tensor,
    V_g: torch.Tensor,
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
    offset_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Newton step for β and variational means with fixed V and diagonal Σ."""
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    device, dtype = Xm.device, Xm.dtype
    active_m = mask_m.bool()
    active_q_f = active_q.to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    prec_q = torch.where(active_q, 1.0 / sigma.clamp(min=1e-4).square(), torch.ones_like(sigma))

    eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
    eta = eta + torch.einsum('bmnq,bmq->bmn', Z_eff, means)
    eta = eta + _poissonVariationalOffset(V_g, Zm, active_q, offset_clip)
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
    score_u = score_u - prec_q[:, None, :] * means
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
    means_new = means + float(damping) * delta_u
    means_new = means_new.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    means_new = means_new.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    means_new = means_new * active_m[:, :, None] * active_q_f[:, None, :]
    return beta_new, means_new, A


def _poissonVariationalTargetDiag(
    beta: torch.Tensor,
    means: torch.Tensor,
    V_g: torch.Tensor,
    sigma: torch.Tensor,
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
    offset_clip: float | None = None,
) -> torch.Tensor:
    """ELBO-style target for q(u_g)=N(m_g,V_g) with diagonal random-effect Σ."""
    B, _, _, d = Xm.shape
    q = Zm.shape[-1]
    dtype = Xm.dtype
    active_q_f = active_q.to(dtype)
    Z_eff = Zm * active_q_f[:, None, None, :]
    sigma = sigma.clamp(min=1e-4)
    sigma2 = sigma.square()

    eta_linear = torch.einsum('bmnd,bd->bmn', Xm, beta)
    eta_linear = eta_linear + torch.einsum('bmnq,bmq->bmn', Z_eff, means)
    eta_mean = eta_linear + _poissonVariationalOffset(V_g, Zm, active_q, offset_clip)
    eta_mean_eff = eta_mean.clamp(max=_POISSON_ETA_CLIP_MAX)
    ll = ym * eta_linear.clamp(min=-30.0, max=_POISSON_ETA_CLIP_MAX) - torch.exp(eta_mean_eff)
    target = (ll * mask_n).sum(dim=(1, 2))

    V_diag = V_g.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    second_moment = means.square() + V_diag
    second_moment = second_moment * mask_m[:, :, None] * active_q_f[:, None, :]
    quad = (second_moment / sigma2[:, None, :].clamp(min=1e-8)).sum(dim=-1)
    log_det_sigma = (2.0 * sigma.log() * active_q_f).sum(dim=-1)
    q_count = active_q_f.sum(dim=-1)
    log_prior_u = -0.5 * (
        q_count[:, None] * math.log(2.0 * math.pi) + log_det_sigma[:, None] + quad
    )
    target = target + (log_prior_u * mask_m).sum(dim=-1)

    inactive_q_f = (~active_q).to(dtype)
    V_safe = V_g * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]
    V_safe = V_safe + torch.diag_embed(inactive_q_f)[:, None]
    sign_v, log_det_v = torch.linalg.slogdet(V_safe)
    log_det_v = torch.where(sign_v > 0, log_det_v, log_det_v.new_zeros(()))
    entropy = 0.5 * (q_count[:, None] * (1.0 + math.log(2.0 * math.pi)) + log_det_v)
    target = target + (entropy * mask_m).sum(dim=-1)

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
        target = target + (sigma.log() * active_q_f).sum(dim=-1)

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


def refinePoissonLaplacePirlsSigmaAverage(
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
    scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.3333333, 2.0),
    scale_mode: str = 'scalar',
    intercept_scales: tuple[float, ...] = (0.75, 1.0, 1.3333333),
    slope_scales: tuple[float, ...] = (0.5, 1.0, 1.5),
    n_steps: int = 2,
    damping: float = 0.5,
    temperature: float = 2.0,
    min_d: int = 1,
    max_q: int | None = None,
    output_mode: str = 'beta',
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Laplace-weighted local integration over diagonal σ scale candidates.

    Each log-σ candidate starts from the current estimate, refreshes β/u with fixed σ using
    a few joint PIRLS steps, and is scored by the diagonal Laplace target.  The primary
    output is the Laplace-weighted β mean; σ/BLUP output is controlled by ``output_mode``:

    - ``'beta'``: write back weighted β only.
    - ``'beta_best'``: weighted β plus σ/BLUPs from the best-scoring candidate.
    - ``'beta_sigma'``: weighted β/σ plus BLUPs from the best-scoring candidate.

    ``scale_mode='scalar'`` applies one multiplier to every active random-effect
    dimension.  ``scale_mode='intercept_slope'`` applies one multiplier to dimension 0
    and one shared multiplier to all remaining dimensions.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_steps <= 0 or 'sigma_rfx_est' not in stats:
        return stats
    if output_mode not in {'beta', 'beta_best', 'beta_sigma'}:
        raise ValueError("output_mode must be 'beta', 'beta_best', or 'beta_sigma'")
    if scale_mode not in {'scalar', 'intercept_slope'}:
        raise ValueError("scale_mode must be 'scalar' or 'intercept_slope'")

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
            out['poisson_sigma_average_gate'] = gate.to(dtype)
            out['poisson_sigma_average_neff'] = torch.ones(B, device=device, dtype=dtype)
            out['poisson_sigma_average_best_scale'] = torch.ones(B, device=device, dtype=dtype)
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

    beta_candidates = [beta_base]
    sigma_candidates = [sigma_base]
    blup_candidates = [blups_base]
    A_candidates = [base_A]
    target_candidates = [base_target]
    scale_candidates = [torch.ones(B, device=device, dtype=dtype)]
    intercept_scale_candidates = [torch.ones(B, device=device, dtype=dtype)]
    slope_scale_candidates = [torch.ones(B, device=device, dtype=dtype)]

    scale_vectors: list[tuple[float, ...]] = []
    if scale_mode == 'scalar':
        for scale in scales:
            scale_f = float(scale)
            scale_vectors.append((scale_f,) * q)
    else:
        for intercept_scale in intercept_scales:
            for slope_scale in slope_scales:
                intercept_f = float(intercept_scale)
                slope_f = float(slope_scale)
                if q == 1:
                    scale_vectors.append((intercept_f,))
                else:
                    scale_vectors.append((intercept_f, *((slope_f,) * (q - 1))))
    scale_vectors = list(dict.fromkeys(scale_vectors))

    for scale_vector in scale_vectors:
        scale_tensor = torch.tensor(scale_vector, device=device, dtype=dtype).expand(B, q)
        sigma_try = torch.where(
            gate[:, None] & active_q,
            (sigma_base * scale_tensor).clamp(min=1e-4, max=sigma_max),
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
        beta_candidates.append(beta_try)
        sigma_candidates.append(sigma_try)
        blup_candidates.append(blups_try)
        A_candidates.append(A_try)
        target_candidates.append(target_try)
        scale_candidates.append(scale_tensor.mean(dim=1))
        intercept_scale_candidates.append(
            torch.full((B,), scale_vector[0], device=device, dtype=dtype)
        )
        slope_scale = scale_vector[1] if len(scale_vector) > 1 else scale_vector[0]
        slope_scale_candidates.append(torch.full((B,), slope_scale, device=device, dtype=dtype))

    beta_stack = torch.stack(beta_candidates, dim=1)
    sigma_stack = torch.stack(sigma_candidates, dim=1)
    blup_stack = torch.stack(blup_candidates, dim=1)
    A_stack = torch.stack(A_candidates, dim=1)
    target_stack = torch.stack(target_candidates, dim=1)
    scale_stack = torch.stack(scale_candidates, dim=1)
    intercept_scale_stack = torch.stack(intercept_scale_candidates, dim=1)
    slope_scale_stack = torch.stack(slope_scale_candidates, dim=1)

    finite_target = torch.isfinite(target_stack)
    score = torch.where(finite_target, target_stack, target_stack.new_full((), -1e30))
    best_idx = score.argmax(dim=1)
    best_target = score.gather(1, best_idx[:, None]).squeeze(1)
    weights = torch.softmax((score - best_target[:, None]) / max(float(temperature), 1e-6), dim=1)
    weights = torch.where(
        gate[:, None], weights, torch.nn.functional.one_hot(best_idx, score.shape[1]).to(dtype)
    )
    neff = 1.0 / weights.square().sum(dim=1).clamp(min=1e-8)

    beta_avg = torch.einsum('bk,bkd->bd', weights, beta_stack)
    sigma_avg = torch.einsum('bk,bkq->bq', weights, sigma_stack).clamp(min=1e-4, max=sigma_max)
    gather_blup = best_idx[:, None, None, None].expand(B, 1, Zm.shape[1], q)
    gather_A = best_idx[:, None, None, None, None].expand(B, 1, Zm.shape[1], q, q)
    best_sigma = sigma_stack.gather(1, best_idx[:, None, None].expand(B, 1, q)).squeeze(1)
    best_blups = blup_stack.gather(1, gather_blup).squeeze(1)
    best_A = A_stack.gather(1, gather_A).squeeze(1)
    best_scale = scale_stack.gather(1, best_idx[:, None]).squeeze(1)
    best_intercept_scale = intercept_scale_stack.gather(1, best_idx[:, None]).squeeze(1)
    best_slope_scale = slope_scale_stack.gather(1, best_idx[:, None]).squeeze(1)

    sigma_out = sigma_base
    blups_out = blups_base
    A_out = base_A
    if output_mode == 'beta_best':
        sigma_out = best_sigma
        blups_out = best_blups
        A_out = best_A
    elif output_mode == 'beta_sigma':
        sigma_out = torch.where(active_q, sigma_avg, sigma_base)
        blups_out = best_blups
        A_out = _poissonJointHessianDiag(
            beta_avg, blups_out, sigma_out, Zm, Xm, mask_n, mask_m, active_q
        )

    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
    H_inv = _safeSolve(A_out + _adaptiveRidgeBm(A_out), eye_q) * mask_m[:, :, None, None]
    blup_var = H_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q.to(dtype)[:, None, :]
    Psi_lap = torch.diag_embed(sigma_out.square())

    out = dict(stats)
    out['beta_est'] = torch.where(gate[:, None], beta_avg, beta_base)
    if output_mode != 'beta':
        out['sigma_rfx_est'] = torch.where(gate[:, None], sigma_out, sigma_base)
        out['blup_est'] = torch.where(gate[:, None, None], blups_out, blups_base)
        if 'blup_var' in stats:
            out['blup_var'] = torch.where(
                gate[:, None, None], blup_var, stats['blup_var'][:, :, :q]
            )
        else:
            out['blup_var'] = blup_var
        out['Psi_lap'] = torch.where(gate[:, None, None], Psi_lap, stats['Psi_lap'][:, :q, :q])
        out['Psi_lap'] = _psdClampEigenvalues(out['Psi_lap'], _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_sigma_average_gate'] = gate.to(dtype)
        out['poisson_sigma_average_neff'] = neff
        out['poisson_sigma_average_best_scale'] = best_scale
        out['poisson_sigma_average_best_intercept_scale'] = best_intercept_scale
        out['poisson_sigma_average_best_slope_scale'] = best_slope_scale
        out['poisson_sigma_average_best_target'] = best_target
        out['poisson_sigma_average_base_target'] = base_target
    return out


def refinePoissonVariationalGaussian(
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
    n_outer: int = 5,
    n_inner: int = 5,
    n_final: int = 2,
    damping: float = 0.7,
    sigma_blend: float = 0.25,
    sigma_prior_weight: float = 4.0,
    adaptive_steps: int = 2,
    adaptive_target_gain: float = 1e-4,
    adaptive_min_beta_step: float = 0.01,
    adaptive_min_mean_step: float = 0.01,
    adaptive_min_offset_step: float = 0.01,
    adaptive_min_sigma_step: float = 0.005,
    offset_clip: float | None = None,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Posterior-mean-oriented Gaussian variational Poisson refinement.

    This prototype uses ``q(u_g)=N(m_g,V_g)`` and optimizes β/m against the expected
    Poisson mean ``exp(Xβ + Zm + 0.5 diag(Z V_g Z'))``.  Diagonal σ is updated from
    posterior second moments ``m_g² + diag(V_g)``.  It is deliberately opt-in until the
    approximation is benchmarked against the retained Laplace/PIRLS path.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_outer <= 0 or 'sigma_rfx_est' not in stats:
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

    beta_base = stats['beta_est'][:, :d].detach().clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    means_base = stats.get('blup_est', Zm.new_zeros(B, Zm.shape[1], q))[:, :, :q].detach()
    means_base = means_base.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    means_base = means_base.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    means_base = means_base * mask_m[:, :, None] * active_q_f[:, None, :]
    sigma_base = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)

    base_A = _poissonJointHessianDiag(
        beta_base, means_base, sigma_base, Zm, Xm, mask_n, mask_m, active_q
    )
    eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
    V_base = _safeSolve(base_A + _adaptiveRidgeBm(base_A), eye_q)
    V_base = V_base * mask_m[:, :, None, None]
    V_base = V_base * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]

    beta = beta_base.clone()
    means = means_base.clone()
    sigma = sigma_base.clone()
    V_g = V_base.clone()
    sigma0 = sigma_base
    if tau_rfx is not None:
        sigma0 = tau_rfx[:, :q].to(device=device, dtype=dtype).clamp(min=1e-4, max=sigma_max)

    A = base_A
    for _ in range(max(int(n_outer), 1)):
        for _ in range(max(int(n_inner), 1)):
            beta, means, _ = _poissonVariationalStepDiag(
                beta,
                means,
                V_g,
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
                offset_clip=offset_clip,
            )
            A, V_g = _poissonVariationalCovarianceDiag(
                beta, means, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )

        sigma = _poissonVariationalSigmaMomentDiag(
            means,
            V_g,
            sigma,
            sigma0,
            mask_m,
            active_q,
            sigma_blend,
            sigma_prior_weight,
            sigma_max,
        )
        A, V_g = _poissonVariationalCovarianceDiag(
            beta, means, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )

    for _ in range(max(int(n_final), 0)):
        beta, means, _ = _poissonVariationalStepDiag(
            beta,
            means,
            V_g,
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
            offset_clip=offset_clip,
        )
        A, V_g = _poissonVariationalCovarianceDiag(
            beta, means, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )

    if n_final > 0:
        sigma = _poissonVariationalSigmaMomentDiag(
            means,
            V_g,
            sigma,
            sigma0,
            mask_m,
            active_q,
            sigma_blend,
            sigma_prior_weight,
            sigma_max,
        )
        A, V_g = _poissonVariationalCovarianceDiag(
            beta, means, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )

    adaptive_accept_count = torch.zeros(B, device=device, dtype=dtype)
    adaptive_beta_step = torch.zeros(B, device=device, dtype=dtype)
    adaptive_mean_step = torch.zeros(B, device=device, dtype=dtype)
    adaptive_offset_step = torch.zeros(B, device=device, dtype=dtype)
    adaptive_sigma_step = torch.zeros(B, device=device, dtype=dtype)
    if adaptive_steps > 0:
        current_target = _poissonVariationalTargetDiag(
            beta,
            means,
            V_g,
            sigma,
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
            offset_clip=offset_clip,
        )
        for _ in range(max(int(adaptive_steps), 0)):
            beta_try, means_try, _ = _poissonVariationalStepDiag(
                beta,
                means,
                V_g,
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
                offset_clip=offset_clip,
            )
            A_try, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )
            sigma_try = _poissonVariationalSigmaMomentDiag(
                means_try,
                V_try,
                sigma,
                sigma0,
                mask_m,
                active_q,
                sigma_blend,
                sigma_prior_weight,
                sigma_max,
            )
            A_try, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_try, sigma_try, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )
            target_try = _poissonVariationalTargetDiag(
                beta_try,
                means_try,
                V_try,
                sigma_try,
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
                offset_clip=offset_clip,
            )

            beta_step = _rowRmsMasked(beta_try - beta, active_d)
            mean_step = _rowRmsMasked(
                means_try - means,
                mask_m[:, :, None].bool() & active_q[:, None, :],
            )
            offset_step = _rowRmsMasked(
                _poissonVariationalOffset(V_try, Zm, active_q, offset_clip)
                - _poissonVariationalOffset(V_g, Zm, active_q, offset_clip),
                mask_n.bool(),
            )
            sigma_step = _rowRmsMasked(
                sigma_try.clamp(min=1e-4).log() - sigma.clamp(min=1e-4).log(),
                active_q,
            )
            moving = (
                (beta_step > float(adaptive_min_beta_step))
                | (mean_step > float(adaptive_min_mean_step))
                | (offset_step > float(adaptive_min_offset_step))
                | (sigma_step > float(adaptive_min_sigma_step))
            )
            improved = torch.isfinite(target_try) & (
                target_try >= current_target + float(adaptive_target_gain)
            )
            accept_adaptive = moving & improved

            beta = torch.where(accept_adaptive[:, None], beta_try, beta)
            means = torch.where(accept_adaptive[:, None, None], means_try, means)
            sigma = torch.where(accept_adaptive[:, None], sigma_try, sigma)
            V_g = torch.where(accept_adaptive[:, None, None, None], V_try, V_g)
            A = torch.where(accept_adaptive[:, None, None, None], A_try, A)
            current_target = torch.where(accept_adaptive, target_try, current_target)
            adaptive_accept_count = adaptive_accept_count + accept_adaptive.to(dtype)
            adaptive_beta_step = torch.where(accept_adaptive, beta_step, adaptive_beta_step)
            adaptive_mean_step = torch.where(accept_adaptive, mean_step, adaptive_mean_step)
            adaptive_offset_step = torch.where(accept_adaptive, offset_step, adaptive_offset_step)
            adaptive_sigma_step = torch.where(accept_adaptive, sigma_step, adaptive_sigma_step)

    final_target = _poissonVariationalTargetDiag(
        beta,
        means,
        V_g,
        sigma,
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
        offset_clip=offset_clip,
    )
    base_target = _poissonVariationalTargetDiag(
        beta_base,
        means_base,
        V_base,
        sigma_base,
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
        offset_clip=offset_clip,
    )
    accept = torch.isfinite(final_target)
    if accept_only_improved:
        accept = accept & (final_target >= base_target - 1e-5)

    blup_var = V_g.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q_f[:, None, :]
    Psi_lap = torch.diag_embed(sigma.square())

    out = dict(stats)
    out['beta_est'] = torch.where(accept[:, None], beta, beta_base)
    out['sigma_rfx_est'] = torch.where(accept[:, None], sigma, sigma_base)
    out['blup_est'] = torch.where(accept[:, None, None], means, means_base)
    if 'blup_var' in stats:
        out['blup_var'] = torch.where(accept[:, None, None], blup_var, stats['blup_var'][:, :, :q])
    else:
        out['blup_var'] = blup_var
    out['Psi_lap'] = torch.where(accept[:, None, None], Psi_lap, stats['Psi_lap'][:, :q, :q])
    out['Psi_lap'] = _psdClampEigenvalues(out['Psi_lap'], _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_variational_gaussian_accept'] = accept.to(dtype)
        out['poisson_variational_gaussian_target'] = final_target
        out['poisson_variational_gaussian_base_target'] = base_target
        out['poisson_variational_gaussian_adaptive_accept_count'] = adaptive_accept_count
        out['poisson_variational_gaussian_adaptive_beta_step'] = adaptive_beta_step
        out['poisson_variational_gaussian_adaptive_mean_step'] = adaptive_mean_step
        out['poisson_variational_gaussian_adaptive_offset_step'] = adaptive_offset_step
        out['poisson_variational_gaussian_adaptive_sigma_step'] = adaptive_sigma_step
    return out


def refinePoissonVariationalGaussianSigmaAverage(
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
    scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.3333333, 2.0),
    scale_mode: str = 'scalar',
    intercept_scales: tuple[float, ...] = (0.75, 1.0, 1.3333333),
    slope_scales: tuple[float, ...] = (0.5, 1.0, 1.5),
    n_steps: int = 2,
    damping: float = 0.7,
    temperature: float = 2.0,
    output_mode: str = 'beta_sigma',
    offset_clip: float | None = None,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """ELBO-weighted local integration over σ candidates around the VG state.

    Each candidate fixes a diagonal σ scale, refreshes β and q(u_g)=N(m_g,V_g) with a
    few variational Newton steps, scores the candidate by the variational target, and
    writes back a posterior-weighted β.  σ can be kept fixed, selected from the best
    candidate, or averaged, while BLUP means are kept from the best-scoring candidate.
    """
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_steps <= 0 or 'sigma_rfx_est' not in stats:
        return stats
    if output_mode not in {'beta', 'beta_best', 'beta_sigma'}:
        raise ValueError("output_mode must be 'beta', 'beta_best', or 'beta_sigma'")
    if scale_mode not in {'scalar', 'intercept_slope'}:
        raise ValueError("scale_mode must be 'scalar' or 'intercept_slope'")

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

    beta_base = stats['beta_est'][:, :d].detach().clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    means_base = stats.get('blup_est', Zm.new_zeros(B, Zm.shape[1], q))[:, :, :q].detach()
    means_base = means_base.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    means_base = means_base.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    means_base = means_base * mask_m[:, :, None] * active_q_f[:, None, :]
    sigma_base = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)

    if 'blup_var' in stats:
        V_base = torch.diag_embed(stats['blup_var'][:, :, :q].detach().clamp(min=0.0, max=25.0))
        V_base = V_base * mask_m[:, :, None, None]
        V_base = V_base * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]
        base_A, V_base = _poissonVariationalCovarianceDiag(
            beta_base, means_base, V_base, sigma_base, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )
    else:
        base_A = _poissonJointHessianDiag(
            beta_base, means_base, sigma_base, Zm, Xm, mask_n, mask_m, active_q
        )
        eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
        V_base = _safeSolve(base_A + _adaptiveRidgeBm(base_A), eye_q)
        V_base = V_base * mask_m[:, :, None, None]
        V_base = V_base * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]

    base_target = _poissonVariationalTargetDiag(
        beta_base,
        means_base,
        V_base,
        sigma_base,
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
        offset_clip=offset_clip,
    )

    beta_candidates = [beta_base]
    sigma_candidates = [sigma_base]
    mean_candidates = [means_base]
    V_candidates = [V_base]
    A_candidates = [base_A]
    target_candidates = [base_target]
    scale_candidates = [torch.ones(B, device=device, dtype=dtype)]
    intercept_scale_candidates = [torch.ones(B, device=device, dtype=dtype)]
    slope_scale_candidates = [torch.ones(B, device=device, dtype=dtype)]

    scale_vectors: list[tuple[float, ...]] = []
    if scale_mode == 'scalar':
        for scale in scales:
            scale_vectors.append((float(scale),) * q)
    else:
        for intercept_scale in intercept_scales:
            for slope_scale in slope_scales:
                if q == 1:
                    scale_vectors.append((float(intercept_scale),))
                else:
                    scale_vectors.append(
                        (float(intercept_scale), *((float(slope_scale),) * (q - 1)))
                    )
    scale_vectors = list(dict.fromkeys(scale_vectors))

    for scale_vector in scale_vectors:
        scale_tensor = torch.tensor(scale_vector, device=device, dtype=dtype).expand(B, q)
        sigma_try = torch.where(
            active_q,
            (sigma_base * scale_tensor).clamp(min=1e-4, max=sigma_max),
            sigma_base,
        )
        beta_try = beta_base.clone()
        means_try = means_base.clone()
        V_try = V_base.clone()
        A_try = base_A
        for _ in range(max(int(n_steps), 1)):
            beta_try, means_try, _ = _poissonVariationalStepDiag(
                beta_try,
                means_try,
                V_try,
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
                offset_clip=offset_clip,
            )
            A_try, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_try, sigma_try, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )

        target_try = _poissonVariationalTargetDiag(
            beta_try,
            means_try,
            V_try,
            sigma_try,
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
            offset_clip=offset_clip,
        )
        beta_candidates.append(beta_try)
        sigma_candidates.append(sigma_try)
        mean_candidates.append(means_try)
        V_candidates.append(V_try)
        A_candidates.append(A_try)
        target_candidates.append(target_try)
        scale_candidates.append(scale_tensor.mean(dim=1))
        intercept_scale_candidates.append(
            torch.full((B,), scale_vector[0], device=device, dtype=dtype)
        )
        slope_scale = scale_vector[1] if len(scale_vector) > 1 else scale_vector[0]
        slope_scale_candidates.append(torch.full((B,), slope_scale, device=device, dtype=dtype))

    beta_stack = torch.stack(beta_candidates, dim=1)
    sigma_stack = torch.stack(sigma_candidates, dim=1)
    mean_stack = torch.stack(mean_candidates, dim=1)
    V_stack = torch.stack(V_candidates, dim=1)
    A_stack = torch.stack(A_candidates, dim=1)
    target_stack = torch.stack(target_candidates, dim=1)
    scale_stack = torch.stack(scale_candidates, dim=1)
    intercept_scale_stack = torch.stack(intercept_scale_candidates, dim=1)
    slope_scale_stack = torch.stack(slope_scale_candidates, dim=1)

    finite_target = torch.isfinite(target_stack)
    score = torch.where(finite_target, target_stack, target_stack.new_full((), -1e30))
    best_idx = score.argmax(dim=1)
    best_target = score.gather(1, best_idx[:, None]).squeeze(1)
    weights = torch.softmax((score - best_target[:, None]) / max(float(temperature), 1e-6), dim=1)
    neff = 1.0 / weights.square().sum(dim=1).clamp(min=1e-8)

    beta_avg = torch.einsum('bk,bkd->bd', weights, beta_stack)
    sigma_avg = torch.einsum('bk,bkq->bq', weights, sigma_stack).clamp(min=1e-4, max=sigma_max)
    gather_mean = best_idx[:, None, None, None].expand(B, 1, Zm.shape[1], q)
    gather_V = best_idx[:, None, None, None, None].expand(B, 1, Zm.shape[1], q, q)
    best_sigma = sigma_stack.gather(1, best_idx[:, None, None].expand(B, 1, q)).squeeze(1)
    best_means = mean_stack.gather(1, gather_mean).squeeze(1)
    best_V = V_stack.gather(1, gather_V).squeeze(1)
    best_A = A_stack.gather(1, gather_V).squeeze(1)
    best_scale = scale_stack.gather(1, best_idx[:, None]).squeeze(1)
    best_intercept_scale = intercept_scale_stack.gather(1, best_idx[:, None]).squeeze(1)
    best_slope_scale = slope_scale_stack.gather(1, best_idx[:, None]).squeeze(1)

    sigma_out = sigma_base
    means_out = means_base
    V_out = V_base
    A_out = base_A
    if output_mode == 'beta_best':
        sigma_out = best_sigma
        means_out = best_means
        V_out = best_V
        A_out = best_A
    elif output_mode == 'beta_sigma':
        sigma_out = torch.where(active_q, sigma_avg, sigma_base)
        means_out = best_means
        _, V_out = _poissonVariationalCovarianceDiag(
            beta_avg, means_out, best_V, sigma_out, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )
        A_out, V_out = _poissonVariationalCovarianceDiag(
            beta_avg, means_out, V_out, sigma_out, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )

    blup_var = V_out.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q_f[:, None, :]
    Psi_lap = torch.diag_embed(sigma_out.square())

    out = dict(stats)
    out['beta_est'] = beta_avg
    if output_mode != 'beta':
        out['sigma_rfx_est'] = sigma_out
        out['blup_est'] = means_out
        if 'blup_var' in stats:
            out['blup_var'] = blup_var
        else:
            out['blup_var'] = blup_var
        out['Psi_lap'] = _psdClampEigenvalues(Psi_lap, _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_vg_sigma_average_neff'] = neff
        out['poisson_vg_sigma_average_best_scale'] = best_scale
        out['poisson_vg_sigma_average_best_intercept_scale'] = best_intercept_scale
        out['poisson_vg_sigma_average_best_slope_scale'] = best_slope_scale
        out['poisson_vg_sigma_average_best_target'] = best_target
        out['poisson_vg_sigma_average_base_target'] = base_target
    return out


def refinePoissonVariationalGaussianStateAverage(
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
    v_scales: tuple[float, ...] = (0.75, 1.0, 1.3333333),
    n_steps: int = 2,
    damping: float = 0.7,
    temperature: float = 2.0,
    output_mode: str = 'beta',
    offset_clip: float | None = None,
    sigma_blend: float = 0.25,
    sigma_prior_weight: float = 4.0,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Average nearby VG fixed points induced by local covariance-temperature scales."""
    d = Xm.shape[-1]
    q = Zm.shape[-1]
    if d == 0 or q == 0 or n_steps <= 0 or 'sigma_rfx_est' not in stats:
        return stats
    if output_mode not in {'beta', 'beta_best', 'beta_sigma'}:
        raise ValueError("output_mode must be 'beta', 'beta_best', or 'beta_sigma'")

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

    beta_base = stats['beta_est'][:, :d].detach().clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    means_base = stats.get('blup_est', Zm.new_zeros(B, Zm.shape[1], q))[:, :, :q].detach()
    means_base = means_base.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    means_base = means_base.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    means_base = means_base * mask_m[:, :, None] * active_q_f[:, None, :]
    sigma_base = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)
    sigma0 = sigma_base
    if tau_rfx is not None:
        sigma0 = tau_rfx[:, :q].to(device=device, dtype=dtype).clamp(min=1e-4, max=sigma_max)

    if 'blup_var' in stats:
        V_base = torch.diag_embed(stats['blup_var'][:, :, :q].detach().clamp(min=0.0, max=25.0))
        V_base = V_base * mask_m[:, :, None, None]
        V_base = V_base * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]
        base_A, V_base = _poissonVariationalCovarianceDiag(
            beta_base, means_base, V_base, sigma_base, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )
    else:
        base_A = _poissonJointHessianDiag(
            beta_base, means_base, sigma_base, Zm, Xm, mask_n, mask_m, active_q
        )
        eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
        V_base = _safeSolve(base_A + _adaptiveRidgeBm(base_A), eye_q)
        V_base = V_base * mask_m[:, :, None, None]
        V_base = V_base * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]

    base_target = _poissonVariationalTargetDiag(
        beta_base,
        means_base,
        V_base,
        sigma_base,
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
        offset_clip=offset_clip,
    )

    beta_candidates = [beta_base]
    sigma_candidates = [sigma_base]
    mean_candidates = [means_base]
    V_candidates = [V_base]
    A_candidates = [base_A]
    target_candidates = [base_target]
    scale_candidates = [torch.ones(B, device=device, dtype=dtype)]

    for v_scale in dict.fromkeys(float(scale) for scale in v_scales):
        if abs(v_scale - 1.0) < 1e-8:
            continue
        beta_try = beta_base.clone()
        means_try = means_base.clone()
        sigma_try = sigma_base.clone()
        V_try = V_base * float(v_scale)
        V_try = V_try * mask_m[:, :, None, None]
        V_try = V_try * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]
        A_try = base_A
        for _ in range(max(int(n_steps), 1)):
            beta_try, means_try, _ = _poissonVariationalStepDiag(
                beta_try,
                means_try,
                V_try,
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
                offset_clip=offset_clip,
            )
            A_try, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_try, sigma_try, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )
            sigma_try = _poissonVariationalSigmaMomentDiag(
                means_try,
                V_try,
                sigma_try,
                sigma0,
                mask_m,
                active_q,
                sigma_blend,
                sigma_prior_weight,
                sigma_max,
            )
            A_try, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_try, sigma_try, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )

        target_try = _poissonVariationalTargetDiag(
            beta_try,
            means_try,
            V_try,
            sigma_try,
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
            offset_clip=offset_clip,
        )
        beta_candidates.append(beta_try)
        sigma_candidates.append(sigma_try)
        mean_candidates.append(means_try)
        V_candidates.append(V_try)
        A_candidates.append(A_try)
        target_candidates.append(target_try)
        scale_candidates.append(torch.full((B,), v_scale, device=device, dtype=dtype))

    beta_stack = torch.stack(beta_candidates, dim=1)
    sigma_stack = torch.stack(sigma_candidates, dim=1)
    mean_stack = torch.stack(mean_candidates, dim=1)
    V_stack = torch.stack(V_candidates, dim=1)
    A_stack = torch.stack(A_candidates, dim=1)
    target_stack = torch.stack(target_candidates, dim=1)
    scale_stack = torch.stack(scale_candidates, dim=1)

    finite_target = torch.isfinite(target_stack)
    score = torch.where(finite_target, target_stack, target_stack.new_full((), -1e30))
    best_idx = score.argmax(dim=1)
    best_target = score.gather(1, best_idx[:, None]).squeeze(1)
    weights = torch.softmax((score - best_target[:, None]) / max(float(temperature), 1e-6), dim=1)
    neff = 1.0 / weights.square().sum(dim=1).clamp(min=1e-8)

    beta_avg = torch.einsum('bk,bkd->bd', weights, beta_stack)
    sigma_avg = torch.einsum('bk,bkq->bq', weights, sigma_stack).clamp(min=1e-4, max=sigma_max)
    gather_mean = best_idx[:, None, None, None].expand(B, 1, Zm.shape[1], q)
    gather_V = best_idx[:, None, None, None, None].expand(B, 1, Zm.shape[1], q, q)
    best_sigma = sigma_stack.gather(1, best_idx[:, None, None].expand(B, 1, q)).squeeze(1)
    best_means = mean_stack.gather(1, gather_mean).squeeze(1)
    best_V = V_stack.gather(1, gather_V).squeeze(1)
    best_A = A_stack.gather(1, gather_V).squeeze(1)
    best_scale = scale_stack.gather(1, best_idx[:, None]).squeeze(1)

    sigma_out = sigma_base
    means_out = means_base
    V_out = V_base
    A_out = base_A
    if output_mode == 'beta_best':
        sigma_out = best_sigma
        means_out = best_means
        V_out = best_V
        A_out = best_A
    elif output_mode == 'beta_sigma':
        sigma_out = torch.where(active_q, sigma_avg, sigma_base)
        means_out = best_means
        A_out, V_out = _poissonVariationalCovarianceDiag(
            beta_avg, means_out, best_V, sigma_out, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )

    blup_var = V_out.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q_f[:, None, :]
    Psi_lap = torch.diag_embed(sigma_out.square())

    out = dict(stats)
    out['beta_est'] = beta_avg
    if output_mode != 'beta':
        out['sigma_rfx_est'] = sigma_out
        out['blup_est'] = means_out
        out['blup_var'] = blup_var
        out['Psi_lap'] = _psdClampEigenvalues(Psi_lap, _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_vg_state_average_neff'] = neff
        out['poisson_vg_state_average_best_scale'] = best_scale
        out['poisson_vg_state_average_best_target'] = best_target
        out['poisson_vg_state_average_base_target'] = base_target
    return out


def refinePoissonVariationalGaussianPolish(
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
    n_steps: int = 2,
    dampings: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    target_gain: float = 1e-4,
    offset_clip: float | None = None,
    sigma_blend: float = 0.25,
    sigma_prior_weight: float = 4.0,
    sigma_max: float = math.sqrt(_POISSON_PSI_EIG_CAP),
    max_beta_step: float = 0.75,
    max_blup_step: float = 1.0,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Target-accepted VG line-search polish over damped β/q(u) Newton steps."""
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
    active_q_f = active_q.to(dtype)

    beta = stats['beta_est'][:, :d].detach().clamp(-_POISSON_BETA_CLAMP, _POISSON_BETA_CLAMP)
    means = stats.get('blup_est', Zm.new_zeros(B, Zm.shape[1], q))[:, :, :q].detach()
    means = means.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    means = means.clamp(-_POISSON_BLUP_CLAMP, _POISSON_BLUP_CLAMP)
    means = means * mask_m[:, :, None] * active_q_f[:, None, :]
    sigma = stats['sigma_rfx_est'][:, :q].detach().clamp(min=1e-4, max=sigma_max)
    sigma0 = sigma
    if tau_rfx is not None:
        sigma0 = tau_rfx[:, :q].to(device=device, dtype=dtype).clamp(min=1e-4, max=sigma_max)

    if 'blup_var' in stats:
        V_g = torch.diag_embed(stats['blup_var'][:, :, :q].detach().clamp(min=0.0, max=25.0))
        V_g = V_g * mask_m[:, :, None, None]
        V_g = V_g * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]
        _, V_g = _poissonVariationalCovarianceDiag(
            beta, means, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
        )
    else:
        A = _poissonJointHessianDiag(beta, means, sigma, Zm, Xm, mask_n, mask_m, active_q)
        eye_q = torch.eye(q, device=device, dtype=dtype).expand(B, Zm.shape[1], q, q)
        V_g = _safeSolve(A + _adaptiveRidgeBm(A), eye_q)
        V_g = V_g * mask_m[:, :, None, None]
        V_g = V_g * active_q_f[:, None, :, None] * active_q_f[:, None, None, :]

    current_target = _poissonVariationalTargetDiag(
        beta,
        means,
        V_g,
        sigma,
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
        offset_clip=offset_clip,
    )
    base_target = current_target
    accept_count = torch.zeros(B, device=device, dtype=dtype)
    best_damping_out = torch.zeros(B, device=device, dtype=dtype)

    for _ in range(max(int(n_steps), 1)):
        beta_candidates = [beta]
        mean_candidates = [means]
        sigma_candidates = [sigma]
        V_candidates = [V_g]
        target_candidates = [current_target]
        damping_candidates = [torch.zeros(B, device=device, dtype=dtype)]

        for damping in dict.fromkeys(float(value) for value in dampings):
            if damping <= 0.0:
                continue
            beta_try, means_try, _ = _poissonVariationalStepDiag(
                beta,
                means,
                V_g,
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
                offset_clip=offset_clip,
            )
            _, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_g, sigma, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )
            sigma_try = _poissonVariationalSigmaMomentDiag(
                means_try,
                V_try,
                sigma,
                sigma0,
                mask_m,
                active_q,
                sigma_blend,
                sigma_prior_weight,
                sigma_max,
            )
            _, V_try = _poissonVariationalCovarianceDiag(
                beta_try, means_try, V_try, sigma_try, Zm, Xm, mask_n, mask_m, active_q, offset_clip
            )
            target_try = _poissonVariationalTargetDiag(
                beta_try,
                means_try,
                V_try,
                sigma_try,
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
                offset_clip=offset_clip,
            )
            beta_candidates.append(beta_try)
            mean_candidates.append(means_try)
            sigma_candidates.append(sigma_try)
            V_candidates.append(V_try)
            target_candidates.append(target_try)
            damping_candidates.append(torch.full((B,), damping, device=device, dtype=dtype))

        target_stack = torch.stack(target_candidates, dim=1)
        finite_target = torch.isfinite(target_stack)
        score = torch.where(finite_target, target_stack, target_stack.new_full((), -1e30))
        best_idx = score.argmax(dim=1)
        best_target = score.gather(1, best_idx[:, None]).squeeze(1)
        improved = best_target >= current_target + float(target_gain)
        gather_beta = best_idx[:, None, None].expand(B, 1, d)
        gather_mean = best_idx[:, None, None, None].expand(B, 1, Zm.shape[1], q)
        gather_V = best_idx[:, None, None, None, None].expand(B, 1, Zm.shape[1], q, q)
        beta_best = torch.stack(beta_candidates, dim=1).gather(1, gather_beta).squeeze(1)
        mean_best = torch.stack(mean_candidates, dim=1).gather(1, gather_mean).squeeze(1)
        sigma_best = (
            torch.stack(sigma_candidates, dim=1)
            .gather(1, best_idx[:, None, None].expand(B, 1, q))
            .squeeze(1)
        )
        V_best = torch.stack(V_candidates, dim=1).gather(1, gather_V).squeeze(1)
        damping_best = (
            torch.stack(damping_candidates, dim=1).gather(1, best_idx[:, None]).squeeze(1)
        )

        beta = torch.where(improved[:, None], beta_best, beta)
        means = torch.where(improved[:, None, None], mean_best, means)
        sigma = torch.where(improved[:, None], sigma_best, sigma)
        V_g = torch.where(improved[:, None, None, None], V_best, V_g)
        current_target = torch.where(improved, best_target, current_target)
        accept_count = accept_count + improved.to(dtype)
        best_damping_out = torch.where(improved, damping_best, best_damping_out)

    blup_var = V_g.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var * mask_m[:, :, None] * active_q_f[:, None, :]
    Psi_lap = torch.diag_embed(sigma.square())

    out = dict(stats)
    out['beta_est'] = beta
    out['sigma_rfx_est'] = sigma
    out['blup_est'] = means
    out['blup_var'] = blup_var
    out['Psi_lap'] = _psdClampEigenvalues(Psi_lap, _POISSON_PSI_EIG_CAP)
    if return_diagnostics:
        out['poisson_vg_polish_accept_count'] = accept_count
        out['poisson_vg_polish_best_damping'] = best_damping_out
        out['poisson_vg_polish_target'] = current_target
        out['poisson_vg_polish_base_target'] = base_target
    return out


def _addPoissonLaplaceEbSkippedDiagnostics(
    stats: dict[str, torch.Tensor],
    gate: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    B = gate.shape[0]
    device = gate.device
    if 'laplace_eb_accept' not in stats:
        stats['laplace_eb_accept'] = torch.zeros(B, device=device, dtype=dtype)
    if 'laplace_eb_steps' not in stats:
        stats['laplace_eb_steps'] = torch.zeros(B, device=device, dtype=dtype)
    if 'laplace_eb_target' not in stats:
        stats['laplace_eb_target'] = torch.full((B,), float('nan'), device=device, dtype=dtype)
    if 'laplace_eb_base_target' not in stats:
        stats['laplace_eb_base_target'] = torch.full((B,), float('nan'), device=device, dtype=dtype)
    if 'laplace_eb_blup_fallback' not in stats:
        stats['laplace_eb_blup_fallback'] = torch.zeros(B, device=device, dtype=dtype)
    if 'laplace_eb_beta_jump' not in stats:
        stats['laplace_eb_beta_jump'] = torch.full((B,), float('nan'), device=device, dtype=dtype)
    stats['laplace_eb_gate'] = gate.to(dtype=dtype)


def refinePoissonPath(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    map_priors: dict[str, torch.Tensor | None],
    mask_d: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    map_refine: bool = True,
    options: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Apply the configured Poisson refinement chain after the PQL initializer."""
    if not map_refine or Zm.shape[-1] == 0:
        return stats

    options = {} if options is None else options
    diagnostics = options.get('poisson_laplace_eb_diagnostics', False)
    prior_kwargs = {
        'nu_ffx': map_priors['nu_ffx'],
        'tau_ffx': map_priors['tau_ffx'],
        'family_ffx': map_priors['family_ffx'],
        'tau_rfx': map_priors['tau_rfx'],
        'family_sigma_rfx': map_priors['family_sigma_rfx'],
    }

    def _run(refiner, include_sigma_prior: bool = True, **kwargs):
        selected_priors = (
            prior_kwargs
            if include_sigma_prior
            else {
                'nu_ffx': map_priors['nu_ffx'],
                'tau_ffx': map_priors['tau_ffx'],
                'family_ffx': map_priors['family_ffx'],
            }
        )
        return refiner(
            stats,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            **selected_priors,
            mask_d=mask_d,
            mask_q=mask_q,
            return_diagnostics=diagnostics,
            **kwargs,
        )

    if options.get('poisson_laplace_eb_mode', 'off') != 'off':
        gate = torch.ones(Xm.shape[0], device=Xm.device, dtype=torch.bool)
        stats = _run(
            refinePoissonLaplaceEb,
            n_steps=options['poisson_laplace_eb_steps'],
            n_inner=options['poisson_laplace_eb_inner'],
            n_final=options['poisson_laplace_eb_final'],
            lr=options['poisson_laplace_eb_lr'],
            blup_fallback_beta_jump=options['poisson_laplace_eb_blup_fallback_beta_jump'],
            sigma_prior_cap=options['poisson_laplace_eb_sigma_prior_cap'],
            sigma_prior_cap_min_d=options['poisson_laplace_eb_sigma_prior_cap_min_d'],
        )
        if diagnostics:
            _addPoissonLaplaceEbSkippedDiagnostics(stats, gate, Xm.dtype)

    if options.get('poisson_laplace_pirls_diag', False):
        stats = _run(
            refinePoissonLaplacePirlsDiag,
            n_outer=options['poisson_laplace_pirls_diag_outer'],
            n_pirls=options['poisson_laplace_pirls_diag_inner'],
            n_final=options['poisson_laplace_pirls_diag_final'],
            damping=options['poisson_laplace_pirls_diag_damping'],
            sigma_blend=options['poisson_laplace_pirls_diag_sigma_blend'],
            sigma_prior_weight=options['poisson_laplace_pirls_diag_prior_weight'],
        )

    if options.get('poisson_marginal_beta', False):
        stats = _run(
            refinePoissonMarginalMeanBeta,
            include_sigma_prior=False,
            n_steps=options['poisson_marginal_beta_steps'],
            damping=options['poisson_marginal_beta_damping'],
            min_d=options['poisson_marginal_beta_min_d'],
            max_q=options['poisson_marginal_beta_max_q'],
            max_step=options['poisson_marginal_beta_max_step'],
        )

    if options.get('poisson_laplace_pirls_sigma_grid', False):
        stats = _run(
            refinePoissonLaplacePirlsSigmaGrid,
            scales=tuple(float(x) for x in options['poisson_laplace_pirls_sigma_grid_scales']),
            n_steps=options['poisson_laplace_pirls_sigma_grid_steps'],
            damping=options['poisson_laplace_pirls_diag_damping'],
            min_d=options['poisson_laplace_pirls_sigma_grid_min_d'],
            max_q=options['poisson_laplace_pirls_sigma_grid_max_q'],
        )

    if options.get('poisson_laplace_pirls_sigma_average', False):
        stats = _run(
            refinePoissonLaplacePirlsSigmaAverage,
            scales=tuple(float(x) for x in options['poisson_laplace_pirls_sigma_average_scales']),
            scale_mode=options['poisson_laplace_pirls_sigma_average_scale_mode'],
            intercept_scales=tuple(
                float(x) for x in options['poisson_laplace_pirls_sigma_average_intercept_scales']
            ),
            slope_scales=tuple(
                float(x) for x in options['poisson_laplace_pirls_sigma_average_slope_scales']
            ),
            n_steps=options['poisson_laplace_pirls_sigma_average_steps'],
            damping=options['poisson_laplace_pirls_diag_damping'],
            temperature=options['poisson_laplace_pirls_sigma_average_temperature'],
            min_d=options['poisson_laplace_pirls_sigma_average_min_d'],
            max_q=options['poisson_laplace_pirls_sigma_average_max_q'],
            output_mode=options['poisson_laplace_pirls_sigma_average_output_mode'],
        )

    if options.get('poisson_variational_gaussian', False):
        stats = _run(
            refinePoissonVariationalGaussian,
            n_outer=options['poisson_variational_gaussian_outer'],
            n_inner=options['poisson_variational_gaussian_inner'],
            n_final=options['poisson_variational_gaussian_final'],
            damping=options['poisson_variational_gaussian_damping'],
            sigma_blend=options['poisson_variational_gaussian_sigma_blend'],
            sigma_prior_weight=options['poisson_variational_gaussian_prior_weight'],
            adaptive_steps=options['poisson_variational_gaussian_adaptive_steps'],
            adaptive_target_gain=options['poisson_variational_gaussian_adaptive_target_gain'],
            adaptive_min_beta_step=options['poisson_variational_gaussian_adaptive_min_beta_step'],
            adaptive_min_mean_step=options['poisson_variational_gaussian_adaptive_min_mean_step'],
            adaptive_min_offset_step=(
                options['poisson_variational_gaussian_adaptive_min_offset_step']
            ),
            adaptive_min_sigma_step=options['poisson_variational_gaussian_adaptive_min_sigma_step'],
            offset_clip=options['poisson_variational_gaussian_offset_clip'],
        )

    if options.get('poisson_variational_gaussian_sigma_average', False):
        stats = _run(
            refinePoissonVariationalGaussianSigmaAverage,
            scales=tuple(
                float(x) for x in options['poisson_variational_gaussian_sigma_average_scales']
            ),
            scale_mode=options['poisson_variational_gaussian_sigma_average_scale_mode'],
            intercept_scales=tuple(
                float(x)
                for x in options['poisson_variational_gaussian_sigma_average_intercept_scales']
            ),
            slope_scales=tuple(
                float(x) for x in options['poisson_variational_gaussian_sigma_average_slope_scales']
            ),
            n_steps=options['poisson_variational_gaussian_sigma_average_steps'],
            damping=options['poisson_variational_gaussian_damping'],
            temperature=options['poisson_variational_gaussian_sigma_average_temperature'],
            output_mode=options['poisson_variational_gaussian_sigma_average_output_mode'],
            offset_clip=options['poisson_variational_gaussian_offset_clip'],
        )

    if options.get('poisson_variational_gaussian_state_average', False):
        stats = _run(
            refinePoissonVariationalGaussianStateAverage,
            v_scales=tuple(
                float(x) for x in options['poisson_variational_gaussian_state_average_v_scales']
            ),
            n_steps=options['poisson_variational_gaussian_state_average_steps'],
            damping=options['poisson_variational_gaussian_damping'],
            temperature=options['poisson_variational_gaussian_state_average_temperature'],
            output_mode=options['poisson_variational_gaussian_state_average_output_mode'],
            sigma_blend=options['poisson_variational_gaussian_sigma_blend'],
            sigma_prior_weight=options['poisson_variational_gaussian_prior_weight'],
            offset_clip=options['poisson_variational_gaussian_offset_clip'],
        )

    if options.get('poisson_variational_gaussian_polish', False):
        stats = _run(
            refinePoissonVariationalGaussianPolish,
            n_steps=options['poisson_variational_gaussian_polish_steps'],
            dampings=tuple(
                float(x) for x in options['poisson_variational_gaussian_polish_dampings']
            ),
            target_gain=options['poisson_variational_gaussian_polish_target_gain'],
            sigma_blend=options['poisson_variational_gaussian_sigma_blend'],
            sigma_prior_weight=options['poisson_variational_gaussian_prior_weight'],
            offset_clip=options['poisson_variational_gaussian_offset_clip'],
        )

    return stats


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
    max_q: int | None = None,
    max_step: float = 1.0,
    accept_only_improved: bool = True,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Gated β update for Poisson marginal means at fixed Ψ.

    The pseudo target uses E[y|β,Ψ] ≈ exp(Xβ + 0.5 diag(ZΨZ')).  By default it uses the
    diagonal EB Ψ. This is not a full marginal likelihood; it is a cheap fixed-Ψ correction
    for rows where conditional PQL/EB β is visibly too extreme relative to INLA.
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
