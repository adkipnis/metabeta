"""Public dispatchers for analytical LMM/GLMM variance-component estimators."""

import torch

from metabeta.analytical.lmm.blup import analyticalBLUPContext
from metabeta.analytical.glmm.bernoulli import (
    refineBernoulliLaplaceEb,
    refineBernoulliMapBeta,
    refineBernoulliNagqSrfx,
    refineBernoulliNestedBeta,
)
from metabeta.analytical.glmm.poisson import (
    refinePoissonLaplaceEb,
    refinePoissonLaplacePirlsDiag,
    refinePoissonLaplacePirlsFull,
    refinePoissonLaplacePirlsSigmaAverage,
    refinePoissonLaplacePirlsSigmaGrid,
    refinePoissonLaplaceTargetRefine,
    refinePoissonMarginalMeanBeta,
    refinePoissonVariationalGaussian,
)
from metabeta.analytical.lmm.map import refineNormalLaplaceEb, refineNormalMapSrfx
from metabeta.analytical.lmm.lmm import lmmNormal
from metabeta.analytical.glmm.pql import lmmBernoulli, lmmPoisson

_MAP_PRIOR_KEYS = (
    'nu_ffx',
    'tau_ffx',
    'family_ffx',
    'tau_rfx',
    'family_sigma_rfx',
    'tau_eps',
    'family_sigma_eps',
)

_BERNOULLI_LAPLACE_EB_DEFAULTS = {
    'bernoulli_laplace_eb_steps': 24,
    'bernoulli_laplace_eb_inner': 4,
    'bernoulli_laplace_eb_final': 8,
    'bernoulli_laplace_eb_lr': 0.05,
    'bernoulli_laplace_eb_beta_output_cap': 3.0,
    'bernoulli_laplace_eb_beta_output_cap_trigger': 8.0,
    'bernoulli_laplace_eb_sigma_prior_cap': 2.5,
    'bernoulli_laplace_eb_sigma_prior_cap_min_d': 5,
}

_POISSON_LAPLACE_EB_DEFAULTS = {
    'poisson_laplace_eb_steps': 24,
    'poisson_laplace_eb_inner': 4,
    'poisson_laplace_eb_final': 6,
    'poisson_laplace_eb_lr': 0.05,
    'poisson_laplace_eb_blup_fallback_beta_jump': 0.0,
    'poisson_laplace_eb_sigma_prior_cap': 2.5,
    'poisson_laplace_eb_sigma_prior_cap_min_d': 5,
}


def _bernoulliLaplaceEbMode(value: bool | str) -> str:
    if isinstance(value, bool):
        return 'all' if value else 'off'
    if isinstance(value, str):
        value = value.lower()
        if value in {'bernoulli_eb', 'calibrated', 'cal', 'default'}:
            return 'cal'
        if value in {'auto', 'gate'}:
            return 'auto'
        if value in {'all', 'true', 'yes', 'on'}:
            return 'all'
        if value in {'off', 'false', 'no'}:
            return 'off'
    raise ValueError(
        "bernoulli_laplace_eb must be bool, 'bernoulli_eb', 'calibrated', 'auto', or 'gate'"
    )


def _bernoulliLaplaceEbKwarg(
    kwargs: dict,
    key: str,
    default,
    preset: dict[str, int | float],
):
    return kwargs.pop(key, preset.get(key, default))


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


def _sliceBatch(
    value: torch.Tensor | None,
    selected: torch.Tensor,
) -> torch.Tensor | None:
    return None if value is None else value[selected]


def _sliceStatsBatch(
    stats: dict[str, torch.Tensor],
    selected: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in stats.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            out[key] = value[selected]
        else:
            out[key] = value
    return out


def _emptyDiagnosticLike(
    key: str,
    value: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    fill_value = float('nan') if key.endswith('target') or key.endswith('beta_jump') else 0.0
    return value.new_full((batch_size, *value.shape[1:]), fill_value)


def _mergeStatsBatch(
    stats: dict[str, torch.Tensor],
    refined: dict[str, torch.Tensor],
    selected: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    out = dict(stats)
    for key, value in refined.items():
        if not torch.is_tensor(value) or value.shape[:1] != (int(selected.sum().item()),):
            out[key] = value
            continue
        if key in out and torch.is_tensor(out[key]) and out[key].shape[:1] == (batch_size,):
            merged = out[key].clone()
            merged[selected] = value
        else:
            merged = _emptyDiagnosticLike(key, value, batch_size)
            merged[selected] = value
        out[key] = merged
    return out


def _addLaplaceEbSkippedDiagnostics(
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


def _bernoulliLaplaceEbGate(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_q: torch.Tensor | None,
    mask_d: torch.Tensor | None,
    min_d: int | None,
    min_sigma: float | None,
    eta_abs: float | None,
) -> torch.Tensor:
    B, _, _, d = Xm.shape
    q = Zm.shape[-1]
    device, dtype = Xm.device, Xm.dtype
    active_q = (
        mask_q[:, :q].to(device=device).bool()
        if mask_q is not None
        else torch.ones(B, q, device=device, dtype=torch.bool)
    )
    has_rfx = active_q.any(dim=1)
    gate = torch.zeros(B, device=device, dtype=torch.bool)

    if min_d is not None:
        if mask_d is None:
            d_eff = torch.full((B,), d, device=device, dtype=torch.long)
        else:
            d_eff = mask_d[:, :d].to(device=device).bool().sum(dim=1)
        gate |= d_eff >= int(min_d)

    if min_sigma is not None and 'sigma_rfx_est' in stats:
        sigma = stats['sigma_rfx_est'][:, :q].detach().to(device=device, dtype=dtype)
        q_count = active_q.to(dtype).sum(dim=1).clamp(min=1.0)
        sigma_mean = (sigma * active_q.to(dtype)).sum(dim=1) / q_count
        gate |= sigma_mean >= float(min_sigma)

    if eta_abs is not None and 'beta_est' in stats and 'blup_est' in stats:
        beta = stats['beta_est'][:, :d].detach().to(device=device, dtype=dtype)
        blup = stats['blup_est'][:, :, :q].detach().to(device=device, dtype=dtype)
        eta = (Xm * beta[:, None, None, :]).sum(dim=-1)
        eta = eta + (Zm[:, :, :, :q] * blup[:, :, None, :]).sum(dim=-1)
        eta_abs_max = eta.abs().masked_fill(~mask_n.to(device=device).bool(), 0.0).amax(dim=(1, 2))
        gate |= eta_abs_max >= float(eta_abs)

    return gate & has_rfx


def glmm(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    likelihood_family: int = 0,
    eta_rfx: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Dispatch to lmmNormal / lmmBernoulli / lmmPoisson by likelihood_family.

    When eta_rfx is provided, datasets with eta_rfx == 0 (uncorrelated rfx) have
    Ψ constrained to diagonal throughout estimation — BLUPs and Ψ outputs are
    consistent with the diagonal constraint.
    """
    map_priors = {key: kwargs.pop(key, None) for key in _MAP_PRIOR_KEYS}
    map_refine = kwargs.pop('map_refine', True)
    map_steps = kwargs.pop('map_steps', 20)
    map_recompute_blup = kwargs.pop('map_recompute_blup', True)
    normal_laplace_eb = kwargs.pop('normal_laplace_eb', likelihood_family == 0)
    normal_laplace_eb_steps = kwargs.pop('normal_laplace_eb_steps', 3)
    normal_laplace_eb_moment_blend = kwargs.pop('normal_laplace_eb_moment_blend', 1.0)
    normal_laplace_eb_prior_weight = kwargs.pop('normal_laplace_eb_prior_weight', 4.0)
    normal_laplace_eb_recompute_blup = kwargs.pop('normal_laplace_eb_recompute_blup', True)
    normal_laplace_eb_sigma_grid_refine = kwargs.pop(
        'normal_laplace_eb_sigma_grid_refine', likelihood_family == 0
    )
    normal_laplace_eb_sigma_grid_scales = kwargs.pop(
        'normal_laplace_eb_sigma_grid_scales', (0.75, 1.0, 1.3333333)
    )
    normal_map_beta_prior_cap = kwargs.pop('normal_map_beta_prior_cap', 4.0)
    normal_beta_sigma_grid = kwargs.pop('normal_beta_sigma_grid', likelihood_family == 0)
    normal_beta_sigma_grid_scales = kwargs.pop(
        'normal_beta_sigma_grid_scales', (0.75, 1.0, 1.3333333)
    )
    normal_beta_sigma_grid_min_d = kwargs.pop('normal_beta_sigma_grid_min_d', 5)
    normal_beta_tail_grid = kwargs.pop('normal_beta_tail_grid', likelihood_family == 0)
    normal_beta_tail_grid_scales = kwargs.pop(
        'normal_beta_tail_grid_scales', (0.75, 1.0, 1.3333333)
    )
    normal_beta_tail_grid_min_d = kwargs.pop('normal_beta_tail_grid_min_d', 9)
    normal_beta_tail_grid_min_cond = kwargs.pop('normal_beta_tail_grid_min_cond', 1000.0)
    normal_beta_tail_grid_blend = kwargs.pop('normal_beta_tail_grid_blend', 0.25)
    beta_alpha_low = kwargs.pop('beta_alpha_low', 0.65)
    beta_alpha_high = kwargs.pop('beta_alpha_high', 0.75)
    bernoulli_laplace_eb_default = 'bernoulli_eb' if likelihood_family == 1 else False
    bernoulli_laplace_eb = kwargs.pop('bernoulli_laplace_eb', bernoulli_laplace_eb_default)
    bernoulli_laplace_eb_mode = _bernoulliLaplaceEbMode(bernoulli_laplace_eb)
    bernoulli_laplace_eb_preset = (
        _BERNOULLI_LAPLACE_EB_DEFAULTS if bernoulli_laplace_eb_mode == 'cal' else {}
    )
    if bernoulli_laplace_eb_mode == 'cal':
        bernoulli_laplace_eb_mode = 'all'
    bernoulli_laplace_eb_diagnostics = kwargs.pop('bernoulli_laplace_eb_diagnostics', False)
    bernoulli_laplace_eb_blup_fallback_beta_jump = kwargs.pop(
        'bernoulli_laplace_eb_blup_fallback_beta_jump', 1.0
    )
    bernoulli_laplace_eb_steps = _bernoulliLaplaceEbKwarg(
        kwargs, 'bernoulli_laplace_eb_steps', 12, bernoulli_laplace_eb_preset
    )
    bernoulli_laplace_eb_inner = _bernoulliLaplaceEbKwarg(
        kwargs, 'bernoulli_laplace_eb_inner', 4, bernoulli_laplace_eb_preset
    )
    bernoulli_laplace_eb_final = _bernoulliLaplaceEbKwarg(
        kwargs, 'bernoulli_laplace_eb_final', 6, bernoulli_laplace_eb_preset
    )
    bernoulli_laplace_eb_lr = _bernoulliLaplaceEbKwarg(
        kwargs, 'bernoulli_laplace_eb_lr', 0.05, bernoulli_laplace_eb_preset
    )
    bernoulli_laplace_eb_beta_output_cap = _bernoulliLaplaceEbKwarg(
        kwargs, 'bernoulli_laplace_eb_beta_output_cap', None, bernoulli_laplace_eb_preset
    )
    bernoulli_laplace_eb_beta_output_cap_trigger = _bernoulliLaplaceEbKwarg(
        kwargs,
        'bernoulli_laplace_eb_beta_output_cap_trigger',
        None,
        bernoulli_laplace_eb_preset,
    )
    bernoulli_laplace_eb_sigma_prior_cap = _bernoulliLaplaceEbKwarg(
        kwargs, 'bernoulli_laplace_eb_sigma_prior_cap', None, bernoulli_laplace_eb_preset
    )
    bernoulli_laplace_eb_sigma_prior_cap_min_d = _bernoulliLaplaceEbKwarg(
        kwargs,
        'bernoulli_laplace_eb_sigma_prior_cap_min_d',
        None,
        bernoulli_laplace_eb_preset,
    )
    bernoulli_laplace_eb_recompute_blup_after_calibration = kwargs.pop(
        'bernoulli_laplace_eb_recompute_blup_after_calibration', True
    )
    bernoulli_laplace_eb_gate_min_d = kwargs.pop('bernoulli_laplace_eb_gate_min_d', 4)
    bernoulli_laplace_eb_gate_min_sigma = kwargs.pop('bernoulli_laplace_eb_gate_min_sigma', 0.75)
    bernoulli_laplace_eb_gate_eta_abs = kwargs.pop('bernoulli_laplace_eb_gate_eta_abs', 8.0)
    poisson_laplace_eb_default = 'poisson_eb' if likelihood_family == 2 else False
    poisson_laplace_eb = kwargs.pop('poisson_laplace_eb', poisson_laplace_eb_default)
    poisson_laplace_eb_mode = _poissonLaplaceEbMode(poisson_laplace_eb)
    poisson_laplace_eb_preset = (
        _POISSON_LAPLACE_EB_DEFAULTS if poisson_laplace_eb_mode == 'cal' else {}
    )
    if poisson_laplace_eb_mode == 'cal':
        poisson_laplace_eb_mode = 'all'
    poisson_laplace_eb_diagnostics = kwargs.pop('poisson_laplace_eb_diagnostics', False)
    poisson_laplace_eb_steps = _bernoulliLaplaceEbKwarg(
        kwargs, 'poisson_laplace_eb_steps', 12, poisson_laplace_eb_preset
    )
    poisson_laplace_eb_inner = _bernoulliLaplaceEbKwarg(
        kwargs, 'poisson_laplace_eb_inner', 4, poisson_laplace_eb_preset
    )
    poisson_laplace_eb_final = _bernoulliLaplaceEbKwarg(
        kwargs, 'poisson_laplace_eb_final', 6, poisson_laplace_eb_preset
    )
    poisson_laplace_eb_lr = _bernoulliLaplaceEbKwarg(
        kwargs, 'poisson_laplace_eb_lr', 0.03, poisson_laplace_eb_preset
    )
    poisson_laplace_eb_blup_fallback_beta_jump = _bernoulliLaplaceEbKwarg(
        kwargs,
        'poisson_laplace_eb_blup_fallback_beta_jump',
        None,
        poisson_laplace_eb_preset,
    )
    poisson_laplace_eb_sigma_prior_cap = _bernoulliLaplaceEbKwarg(
        kwargs, 'poisson_laplace_eb_sigma_prior_cap', None, poisson_laplace_eb_preset
    )
    poisson_laplace_eb_sigma_prior_cap_min_d = _bernoulliLaplaceEbKwarg(
        kwargs,
        'poisson_laplace_eb_sigma_prior_cap_min_d',
        None,
        poisson_laplace_eb_preset,
    )
    poisson_laplace_pirls_diag = kwargs.pop('poisson_laplace_pirls_diag', likelihood_family == 2)
    poisson_laplace_pirls_diag_outer = kwargs.pop('poisson_laplace_pirls_diag_outer', 4)
    poisson_laplace_pirls_diag_inner = kwargs.pop('poisson_laplace_pirls_diag_inner', 1)
    poisson_laplace_pirls_diag_final = kwargs.pop('poisson_laplace_pirls_diag_final', 2)
    poisson_laplace_pirls_diag_damping = kwargs.pop('poisson_laplace_pirls_diag_damping', 0.5)
    poisson_laplace_pirls_diag_sigma_blend = kwargs.pop(
        'poisson_laplace_pirls_diag_sigma_blend', 0.5
    )
    poisson_laplace_pirls_diag_prior_weight = kwargs.pop(
        'poisson_laplace_pirls_diag_prior_weight', 4.0
    )
    poisson_laplace_pirls_full = kwargs.pop('poisson_laplace_pirls_full', False)
    poisson_laplace_pirls_full_psi_blend = kwargs.pop('poisson_laplace_pirls_full_psi_blend', 0.5)
    poisson_laplace_pirls_full_prior_weight = kwargs.pop(
        'poisson_laplace_pirls_full_prior_weight', 4.0
    )
    poisson_laplace_pirls_full_offdiag_shrink = kwargs.pop(
        'poisson_laplace_pirls_full_offdiag_shrink', 0.0
    )
    poisson_laplace_pirls_sigma_grid = kwargs.pop(
        'poisson_laplace_pirls_sigma_grid', bool(poisson_laplace_pirls_diag)
    )
    poisson_laplace_pirls_sigma_grid_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_grid_scales', (0.5, 0.75, 1.0)
    )
    poisson_laplace_pirls_sigma_grid_steps = kwargs.pop('poisson_laplace_pirls_sigma_grid_steps', 2)
    poisson_laplace_pirls_sigma_grid_min_d = kwargs.pop('poisson_laplace_pirls_sigma_grid_min_d', 1)
    poisson_laplace_pirls_sigma_grid_max_q = kwargs.pop(
        'poisson_laplace_pirls_sigma_grid_max_q', None
    )
    poisson_laplace_pirls_sigma_average = kwargs.pop(
        'poisson_laplace_pirls_sigma_average', bool(poisson_laplace_pirls_diag)
    )
    poisson_laplace_pirls_sigma_average_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_scales', (0.5, 0.75, 1.0, 1.3333333, 2.0)
    )
    poisson_laplace_pirls_sigma_average_scale_mode = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_scale_mode', 'scalar'
    )
    poisson_laplace_pirls_sigma_average_intercept_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_intercept_scales', (0.75, 1.0, 1.3333333)
    )
    poisson_laplace_pirls_sigma_average_slope_scales = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_slope_scales', (0.5, 1.0, 1.5)
    )
    poisson_laplace_pirls_sigma_average_steps = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_steps', 2
    )
    poisson_laplace_pirls_sigma_average_temperature = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_temperature', 2.0
    )
    poisson_laplace_pirls_sigma_average_min_d = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_min_d', 1
    )
    poisson_laplace_pirls_sigma_average_max_q = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_max_q', None
    )
    poisson_laplace_pirls_sigma_average_output_mode = kwargs.pop(
        'poisson_laplace_pirls_sigma_average_output_mode', 'beta_sigma'
    )
    poisson_laplace_target_refine = kwargs.pop('poisson_laplace_target_refine', False)
    poisson_laplace_target_refine_steps = kwargs.pop('poisson_laplace_target_refine_steps', 1)
    poisson_laplace_target_refine_lr = kwargs.pop('poisson_laplace_target_refine_lr', 0.02)
    poisson_laplace_target_refine_min_d = kwargs.pop('poisson_laplace_target_refine_min_d', 1)
    poisson_laplace_target_refine_max_q = kwargs.pop('poisson_laplace_target_refine_max_q', None)
    poisson_variational_gaussian = kwargs.pop('poisson_variational_gaussian', False)
    poisson_variational_gaussian_outer = kwargs.pop('poisson_variational_gaussian_outer', 5)
    poisson_variational_gaussian_inner = kwargs.pop('poisson_variational_gaussian_inner', 2)
    poisson_variational_gaussian_final = kwargs.pop('poisson_variational_gaussian_final', 1)
    poisson_variational_gaussian_damping = kwargs.pop('poisson_variational_gaussian_damping', 0.5)
    poisson_variational_gaussian_sigma_blend = kwargs.pop(
        'poisson_variational_gaussian_sigma_blend', 0.25
    )
    poisson_variational_gaussian_prior_weight = kwargs.pop(
        'poisson_variational_gaussian_prior_weight', 4.0
    )
    poisson_marginal_beta = kwargs.pop('poisson_marginal_beta', likelihood_family == 2)
    poisson_marginal_beta_full_psi = kwargs.pop('poisson_marginal_beta_full_psi', False)
    poisson_marginal_beta_full_psi_min_q = kwargs.pop('poisson_marginal_beta_full_psi_min_q', 3)
    poisson_marginal_beta_steps = kwargs.pop('poisson_marginal_beta_steps', 4)
    poisson_marginal_beta_damping = kwargs.pop('poisson_marginal_beta_damping', 0.7)
    poisson_marginal_beta_min_d = kwargs.pop('poisson_marginal_beta_min_d', 1)
    poisson_marginal_beta_max_q = kwargs.pop('poisson_marginal_beta_max_q', None)
    poisson_marginal_beta_max_step = kwargs.pop('poisson_marginal_beta_max_step', 1.0)
    poisson_sigma_grid = kwargs.pop('poisson_sigma_grid', False)
    kwargs.pop('poisson_sigma_grid_scales', None)
    kwargs.pop('poisson_sigma_grid_min_d', None)
    kwargs.pop('poisson_sigma_grid_max_q', None)
    kwargs.pop('poisson_sigma_grid_agq_k', None)
    poisson_agq_beta = kwargs.pop('poisson_agq_beta', False)
    kwargs.pop('poisson_agq_beta_steps', None)
    kwargs.pop('poisson_agq_beta_lr', None)
    kwargs.pop('poisson_agq_beta_min_d', None)
    kwargs.pop('poisson_agq_beta_max_q', None)
    kwargs.pop('poisson_agq_beta_max_step', None)
    kwargs.pop('poisson_agq_beta_agq_k', None)
    if poisson_sigma_grid:
        raise ValueError(
            'poisson_sigma_grid was retired; use poisson_laplace_pirls_sigma_grid instead'
        )
    if poisson_agq_beta:
        raise ValueError('poisson_agq_beta was retired; use poisson_laplace_pirls_sigma_grid')
    mask_d = kwargs.pop('mask_d', None)
    uncorr = (eta_rfx == 0) if eta_rfx is not None else None  # (B,) bool or None
    if likelihood_family == 0:
        stats = lmmNormal(
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            ns,
            n_total,
            uncorr=uncorr,
            mask_q=mask_q,
            beta_alpha_low=beta_alpha_low,
            beta_alpha_high=beta_alpha_high,
        )
        if map_refine and Zm.shape[-1] > 0 and all(v is not None for v in map_priors.values()):
            stats = refineNormalMapSrfx(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                ns,
                map_priors['nu_ffx'],
                map_priors['tau_ffx'],
                map_priors['family_ffx'],
                map_priors['tau_rfx'],
                map_priors['family_sigma_rfx'],
                map_priors['tau_eps'],
                map_priors['family_sigma_eps'],
                eta_rfx=eta_rfx,
                mask_d=mask_d,
                mask_q=mask_q,
                n_steps=map_steps,
                recompute_blup=map_recompute_blup,
                beta_alpha_low=beta_alpha_low,
                beta_alpha_high=beta_alpha_high,
                beta_prior_cap=normal_map_beta_prior_cap,
                beta_sigma_grid=normal_beta_sigma_grid,
                beta_sigma_grid_scales=normal_beta_sigma_grid_scales,
                beta_sigma_grid_min_d=normal_beta_sigma_grid_min_d,
            )
        if (
            map_refine
            and normal_laplace_eb
            and Zm.shape[-1] > 0
            and all(v is not None for v in map_priors.values())
        ):
            stats = refineNormalLaplaceEb(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                ns,
                map_priors['nu_ffx'],
                map_priors['tau_ffx'],
                map_priors['family_ffx'],
                map_priors['tau_rfx'],
                map_priors['family_sigma_rfx'],
                map_priors['tau_eps'],
                map_priors['family_sigma_eps'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_steps=normal_laplace_eb_steps,
                moment_blend=normal_laplace_eb_moment_blend,
                prior_weight=normal_laplace_eb_prior_weight,
                recompute_blup=normal_laplace_eb_recompute_blup,
                beta_alpha_low=beta_alpha_low,
                beta_alpha_high=beta_alpha_high,
                sigma_grid_refine=normal_laplace_eb_sigma_grid_refine,
                sigma_grid_scales=normal_laplace_eb_sigma_grid_scales,
                beta_tail_grid=normal_beta_tail_grid,
                beta_tail_grid_scales=normal_beta_tail_grid_scales,
                beta_tail_grid_min_d=normal_beta_tail_grid_min_d,
                beta_tail_grid_min_cond=normal_beta_tail_grid_min_cond,
                beta_tail_grid_blend=normal_beta_tail_grid_blend,
            )
    elif likelihood_family == 1:
        stats = lmmBernoulli(
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            ns,
            n_total,
            uncorr=uncorr,
            nu_ffx=map_priors['nu_ffx'] if map_refine else None,
            tau_ffx=map_priors['tau_ffx'] if map_refine else None,
            family_ffx=map_priors['family_ffx'] if map_refine else None,
            tau_rfx=map_priors['tau_rfx'] if map_refine else None,
            **kwargs,
        )
        if map_refine and Zm.shape[-1] > 0:
            stats = refineBernoulliNagqSrfx(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                mask_q=mask_q,
            )
        if map_refine:
            stats = refineBernoulliNestedBeta(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
            )
        if map_refine and bernoulli_laplace_eb_mode != 'off' and Zm.shape[-1] > 0:
            if bernoulli_laplace_eb_mode == 'all':
                gate = torch.ones(Xm.shape[0], device=Xm.device, dtype=torch.bool)
                stats = refineBernoulliLaplaceEb(
                    stats,
                    Xm,
                    ym,
                    Zm,
                    mask_n,
                    mask_m,
                    nu_ffx=map_priors['nu_ffx'],
                    tau_ffx=map_priors['tau_ffx'],
                    family_ffx=map_priors['family_ffx'],
                    tau_rfx=map_priors['tau_rfx'],
                    family_sigma_rfx=map_priors['family_sigma_rfx'],
                    mask_d=mask_d,
                    mask_q=mask_q,
                    n_steps=bernoulli_laplace_eb_steps,
                    n_inner=bernoulli_laplace_eb_inner,
                    n_final=bernoulli_laplace_eb_final,
                    lr=bernoulli_laplace_eb_lr,
                    blup_fallback_beta_jump=bernoulli_laplace_eb_blup_fallback_beta_jump,
                    beta_output_cap=bernoulli_laplace_eb_beta_output_cap,
                    beta_output_cap_trigger=bernoulli_laplace_eb_beta_output_cap_trigger,
                    sigma_prior_cap=bernoulli_laplace_eb_sigma_prior_cap,
                    sigma_prior_cap_min_d=bernoulli_laplace_eb_sigma_prior_cap_min_d,
                    recompute_blup_after_calibration=(
                        bernoulli_laplace_eb_recompute_blup_after_calibration
                    ),
                    return_diagnostics=bernoulli_laplace_eb_diagnostics,
                )
            else:
                gate = _bernoulliLaplaceEbGate(
                    stats,
                    Xm,
                    Zm,
                    mask_n,
                    mask_q,
                    mask_d,
                    bernoulli_laplace_eb_gate_min_d,
                    bernoulli_laplace_eb_gate_min_sigma,
                    bernoulli_laplace_eb_gate_eta_abs,
                )
                if gate.any():
                    refined = refineBernoulliLaplaceEb(
                        _sliceStatsBatch(stats, gate, Xm.shape[0]),
                        Xm[gate],
                        ym[gate],
                        Zm[gate],
                        mask_n[gate],
                        mask_m[gate],
                        nu_ffx=_sliceBatch(map_priors['nu_ffx'], gate),
                        tau_ffx=_sliceBatch(map_priors['tau_ffx'], gate),
                        family_ffx=_sliceBatch(map_priors['family_ffx'], gate),
                        tau_rfx=_sliceBatch(map_priors['tau_rfx'], gate),
                        family_sigma_rfx=_sliceBatch(map_priors['family_sigma_rfx'], gate),
                        mask_d=_sliceBatch(mask_d, gate),
                        mask_q=_sliceBatch(mask_q, gate),
                        n_steps=bernoulli_laplace_eb_steps,
                        n_inner=bernoulli_laplace_eb_inner,
                        n_final=bernoulli_laplace_eb_final,
                        lr=bernoulli_laplace_eb_lr,
                        blup_fallback_beta_jump=bernoulli_laplace_eb_blup_fallback_beta_jump,
                        beta_output_cap=bernoulli_laplace_eb_beta_output_cap,
                        beta_output_cap_trigger=bernoulli_laplace_eb_beta_output_cap_trigger,
                        sigma_prior_cap=bernoulli_laplace_eb_sigma_prior_cap,
                        sigma_prior_cap_min_d=bernoulli_laplace_eb_sigma_prior_cap_min_d,
                        recompute_blup_after_calibration=(
                            bernoulli_laplace_eb_recompute_blup_after_calibration
                        ),
                        return_diagnostics=bernoulli_laplace_eb_diagnostics,
                    )
                    stats = _mergeStatsBatch(stats, refined, gate, Xm.shape[0])
            if bernoulli_laplace_eb_diagnostics:
                _addLaplaceEbSkippedDiagnostics(stats, gate, Xm.dtype)
    elif likelihood_family == 2:
        stats = lmmPoisson(Xm, ym, Zm, mask_n, mask_m, ns, n_total, uncorr=uncorr, **kwargs)
        if map_refine and poisson_laplace_eb_mode != 'off' and Zm.shape[-1] > 0:
            gate = torch.ones(Xm.shape[0], device=Xm.device, dtype=torch.bool)
            stats = refinePoissonLaplaceEb(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_steps=poisson_laplace_eb_steps,
                n_inner=poisson_laplace_eb_inner,
                n_final=poisson_laplace_eb_final,
                lr=poisson_laplace_eb_lr,
                blup_fallback_beta_jump=poisson_laplace_eb_blup_fallback_beta_jump,
                sigma_prior_cap=poisson_laplace_eb_sigma_prior_cap,
                sigma_prior_cap_min_d=poisson_laplace_eb_sigma_prior_cap_min_d,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
            if poisson_laplace_eb_diagnostics:
                _addLaplaceEbSkippedDiagnostics(stats, gate, Xm.dtype)
        if map_refine and poisson_laplace_pirls_diag and Zm.shape[-1] > 0:
            stats = refinePoissonLaplacePirlsDiag(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_outer=poisson_laplace_pirls_diag_outer,
                n_pirls=poisson_laplace_pirls_diag_inner,
                n_final=poisson_laplace_pirls_diag_final,
                damping=poisson_laplace_pirls_diag_damping,
                sigma_blend=poisson_laplace_pirls_diag_sigma_blend,
                sigma_prior_weight=poisson_laplace_pirls_diag_prior_weight,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
        if map_refine and poisson_laplace_pirls_full and Zm.shape[-1] > 0:
            stats = refinePoissonLaplacePirlsFull(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_outer=poisson_laplace_pirls_diag_outer,
                n_pirls=poisson_laplace_pirls_diag_inner,
                n_final=poisson_laplace_pirls_diag_final,
                damping=poisson_laplace_pirls_diag_damping,
                psi_blend=poisson_laplace_pirls_full_psi_blend,
                psi_prior_weight=poisson_laplace_pirls_full_prior_weight,
                offdiag_shrink=poisson_laplace_pirls_full_offdiag_shrink,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
        if map_refine and poisson_marginal_beta and Zm.shape[-1] > 0:
            poisson_marginal_Psi_lap = (
                stats['Psi_lap'].detach() if poisson_marginal_beta_full_psi else None
            )
            stats = refinePoissonMarginalMeanBeta(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_steps=poisson_marginal_beta_steps,
                damping=poisson_marginal_beta_damping,
                min_d=poisson_marginal_beta_min_d,
                max_q=poisson_marginal_beta_max_q,
                max_step=poisson_marginal_beta_max_step,
                marginal_psi_lap=poisson_marginal_Psi_lap,
                full_psi_min_q=poisson_marginal_beta_full_psi_min_q,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
        if map_refine and poisson_laplace_pirls_sigma_grid and Zm.shape[-1] > 0:
            stats = refinePoissonLaplacePirlsSigmaGrid(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                scales=tuple(float(x) for x in poisson_laplace_pirls_sigma_grid_scales),
                n_steps=poisson_laplace_pirls_sigma_grid_steps,
                damping=poisson_laplace_pirls_diag_damping,
                min_d=poisson_laplace_pirls_sigma_grid_min_d,
                max_q=poisson_laplace_pirls_sigma_grid_max_q,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
        if map_refine and poisson_laplace_pirls_sigma_average and Zm.shape[-1] > 0:
            stats = refinePoissonLaplacePirlsSigmaAverage(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                scales=tuple(float(x) for x in poisson_laplace_pirls_sigma_average_scales),
                scale_mode=poisson_laplace_pirls_sigma_average_scale_mode,
                intercept_scales=tuple(
                    float(x) for x in poisson_laplace_pirls_sigma_average_intercept_scales
                ),
                slope_scales=tuple(
                    float(x) for x in poisson_laplace_pirls_sigma_average_slope_scales
                ),
                n_steps=poisson_laplace_pirls_sigma_average_steps,
                damping=poisson_laplace_pirls_diag_damping,
                temperature=poisson_laplace_pirls_sigma_average_temperature,
                min_d=poisson_laplace_pirls_sigma_average_min_d,
                max_q=poisson_laplace_pirls_sigma_average_max_q,
                output_mode=poisson_laplace_pirls_sigma_average_output_mode,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
        if map_refine and poisson_variational_gaussian and Zm.shape[-1] > 0:
            stats = refinePoissonVariationalGaussian(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_outer=poisson_variational_gaussian_outer,
                n_inner=poisson_variational_gaussian_inner,
                n_final=poisson_variational_gaussian_final,
                damping=poisson_variational_gaussian_damping,
                sigma_blend=poisson_variational_gaussian_sigma_blend,
                sigma_prior_weight=poisson_variational_gaussian_prior_weight,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
        if map_refine and poisson_laplace_target_refine and Zm.shape[-1] > 0:
            stats = refinePoissonLaplaceTargetRefine(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
                nu_ffx=map_priors['nu_ffx'],
                tau_ffx=map_priors['tau_ffx'],
                family_ffx=map_priors['family_ffx'],
                tau_rfx=map_priors['tau_rfx'],
                family_sigma_rfx=map_priors['family_sigma_rfx'],
                mask_d=mask_d,
                mask_q=mask_q,
                n_steps=poisson_laplace_target_refine_steps,
                lr=poisson_laplace_target_refine_lr,
                min_d=poisson_laplace_target_refine_min_d,
                max_q=poisson_laplace_target_refine_max_q,
                return_diagnostics=poisson_laplace_eb_diagnostics,
            )
    else:
        raise ValueError(f'unsupported likelihood_family={likelihood_family}')

    return stats


__all__ = [
    'analyticalBLUPContext',
    'glmm',
    'lmmBernoulli',
    'lmmNormal',
    'lmmPoisson',
    'refineBernoulliLaplaceEb',
    'refineBernoulliMapBeta',
    'refineBernoulliNestedBeta',
    'refineNormalLaplaceEb',
    'refineNormalMapSrfx',
    'refinePoissonLaplaceEb',
    'refinePoissonLaplacePirlsDiag',
    'refinePoissonLaplacePirlsFull',
    'refinePoissonLaplacePirlsSigmaAverage',
    'refinePoissonLaplaceTargetRefine',
    'refinePoissonMarginalMeanBeta',
    'refinePoissonVariationalGaussian',
]
