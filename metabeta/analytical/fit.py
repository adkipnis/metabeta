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
    popPoissonRefinementOptions,
    refinePoissonLaplaceEb,
    refinePoissonLaplacePirlsDiag,
    refinePoissonLaplacePirlsSigmaAverage,
    refinePoissonLaplacePirlsSigmaGrid,
    refinePoissonMarginalMeanBeta,
    refinePoissonPath,
    refinePoissonVariationalGaussian,
    refinePoissonVariationalGaussianSigmaAverage,
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


def _popNormalRefinementOptions(kwargs: dict, likelihood_family: int) -> dict:
    """Pop Normal-path refinement options from ``kwargs``."""
    return {
        'normal_laplace_eb': kwargs.pop('normal_laplace_eb', likelihood_family == 0),
        'normal_laplace_eb_steps': kwargs.pop('normal_laplace_eb_steps', 3),
        'normal_laplace_eb_moment_blend': kwargs.pop('normal_laplace_eb_moment_blend', 1.0),
        'normal_laplace_eb_prior_weight': kwargs.pop('normal_laplace_eb_prior_weight', 4.0),
        'normal_laplace_eb_recompute_blup': kwargs.pop('normal_laplace_eb_recompute_blup', True),
        'normal_laplace_eb_sigma_grid_refine': kwargs.pop(
            'normal_laplace_eb_sigma_grid_refine', likelihood_family == 0
        ),
        'normal_laplace_eb_sigma_grid_scales': kwargs.pop(
            'normal_laplace_eb_sigma_grid_scales', (0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0)
        ),
        'normal_map_beta_prior_cap': kwargs.pop('normal_map_beta_prior_cap', 4.0),
        'normal_map_use_newton': kwargs.pop('normal_map_use_newton', False),
        'normal_map_outer_iterations': kwargs.pop('normal_map_outer_iterations', 2),
        'normal_beta_sigma_grid': kwargs.pop('normal_beta_sigma_grid', likelihood_family == 0),
        'normal_beta_sigma_grid_scales': kwargs.pop(
            'normal_beta_sigma_grid_scales', (0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0)
        ),
        'normal_beta_sigma_grid_min_d': kwargs.pop('normal_beta_sigma_grid_min_d', 5),
        'normal_beta_sigma_grid_unconditional': kwargs.pop(
            'normal_beta_sigma_grid_unconditional', False
        ),
        'normal_beta_sigma_grid_cartesian': kwargs.pop('normal_beta_sigma_grid_cartesian', False),
        'normal_beta_tail_grid': kwargs.pop('normal_beta_tail_grid', likelihood_family == 0),
        'normal_beta_tail_grid_scales': kwargs.pop(
            'normal_beta_tail_grid_scales', (0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0)
        ),
        'normal_beta_tail_grid_min_d': kwargs.pop('normal_beta_tail_grid_min_d', 9),
        'normal_beta_tail_grid_min_cond': kwargs.pop('normal_beta_tail_grid_min_cond', 1000.0),
        'normal_beta_tail_grid_blend': kwargs.pop('normal_beta_tail_grid_blend', 0.25),
        'normal_beta_tail_grid_unconditional': kwargs.pop(
            'normal_beta_tail_grid_unconditional', False
        ),
        'normal_beta_tail_grid_cartesian': kwargs.pop('normal_beta_tail_grid_cartesian', False),
        'beta_alpha_low': kwargs.pop('beta_alpha_low', 0.65),
        'beta_alpha_high': kwargs.pop('beta_alpha_high', 0.75),
    }


def _popBernoulliRefinementOptions(kwargs: dict, likelihood_family: int) -> dict:
    """Pop Bernoulli-path refinement options from ``kwargs``."""
    bernoulli_laplace_eb_default = 'bernoulli_eb' if likelihood_family == 1 else False
    bernoulli_laplace_eb = kwargs.pop('bernoulli_laplace_eb', bernoulli_laplace_eb_default)
    mode = _bernoulliLaplaceEbMode(bernoulli_laplace_eb)
    preset = _BERNOULLI_LAPLACE_EB_DEFAULTS if mode == 'cal' else {}
    if mode == 'cal':
        mode = 'all'
    return {
        'bernoulli_laplace_eb_mode': mode,
        'bernoulli_laplace_eb_diagnostics': kwargs.pop('bernoulli_laplace_eb_diagnostics', False),
        'bernoulli_laplace_eb_blup_fallback_beta_jump': kwargs.pop(
            'bernoulli_laplace_eb_blup_fallback_beta_jump', 1.0
        ),
        'bernoulli_laplace_eb_steps': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_steps', 12, preset
        ),
        'bernoulli_laplace_eb_inner': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_inner', 4, preset
        ),
        'bernoulli_laplace_eb_final': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_final', 6, preset
        ),
        'bernoulli_laplace_eb_lr': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_lr', 0.05, preset
        ),
        'bernoulli_laplace_eb_beta_output_cap': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_beta_output_cap', None, preset
        ),
        'bernoulli_laplace_eb_beta_output_cap_trigger': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_beta_output_cap_trigger', None, preset
        ),
        'bernoulli_laplace_eb_sigma_prior_cap': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_sigma_prior_cap', None, preset
        ),
        'bernoulli_laplace_eb_sigma_prior_cap_min_d': _bernoulliLaplaceEbKwarg(
            kwargs, 'bernoulli_laplace_eb_sigma_prior_cap_min_d', None, preset
        ),
        'bernoulli_laplace_eb_recompute_blup_after_calibration': kwargs.pop(
            'bernoulli_laplace_eb_recompute_blup_after_calibration', True
        ),
        'bernoulli_laplace_eb_gate_min_d': kwargs.pop('bernoulli_laplace_eb_gate_min_d', 4),
        'bernoulli_laplace_eb_gate_min_sigma': kwargs.pop(
            'bernoulli_laplace_eb_gate_min_sigma', 0.75
        ),
        'bernoulli_laplace_eb_gate_eta_abs': kwargs.pop('bernoulli_laplace_eb_gate_eta_abs', 8.0),
    }


def _callBernoulliLaplaceEb(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    map_priors: dict,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    beb: dict,
) -> dict[str, torch.Tensor]:
    return refineBernoulliLaplaceEb(
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
        n_steps=beb['bernoulli_laplace_eb_steps'],
        n_inner=beb['bernoulli_laplace_eb_inner'],
        n_final=beb['bernoulli_laplace_eb_final'],
        lr=beb['bernoulli_laplace_eb_lr'],
        blup_fallback_beta_jump=beb['bernoulli_laplace_eb_blup_fallback_beta_jump'],
        beta_output_cap=beb['bernoulli_laplace_eb_beta_output_cap'],
        beta_output_cap_trigger=beb['bernoulli_laplace_eb_beta_output_cap_trigger'],
        sigma_prior_cap=beb['bernoulli_laplace_eb_sigma_prior_cap'],
        sigma_prior_cap_min_d=beb['bernoulli_laplace_eb_sigma_prior_cap_min_d'],
        recompute_blup_after_calibration=beb[
            'bernoulli_laplace_eb_recompute_blup_after_calibration'
        ],
        return_diagnostics=beb['bernoulli_laplace_eb_diagnostics'],
    )


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
    normal_opts = _popNormalRefinementOptions(kwargs, likelihood_family)
    beb = _popBernoulliRefinementOptions(kwargs, likelihood_family)
    poisson_refinement_options = popPoissonRefinementOptions(kwargs, likelihood_family)
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
            beta_alpha_low=normal_opts['beta_alpha_low'],
            beta_alpha_high=normal_opts['beta_alpha_high'],
        )
        n_outer = int(normal_opts['normal_map_outer_iterations'])
        _priors_ok = Zm.shape[-1] > 0 and all(v is not None for v in map_priors.values())
        _blup_beta_anchor: torch.Tensor | None = None
        for _outer_i in range(n_outer):
            _is_last = _outer_i == n_outer - 1
            if map_refine and _priors_ok:
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
                    beta_alpha_low=normal_opts['beta_alpha_low'],
                    beta_alpha_high=normal_opts['beta_alpha_high'],
                    beta_prior_cap=normal_opts['normal_map_beta_prior_cap'],
                    beta_sigma_grid=normal_opts['normal_beta_sigma_grid'],
                    beta_sigma_grid_scales=normal_opts['normal_beta_sigma_grid_scales'],
                    beta_sigma_grid_min_d=normal_opts['normal_beta_sigma_grid_min_d'],
                    beta_sigma_grid_unconditional=normal_opts['normal_beta_sigma_grid_unconditional'],
                    beta_sigma_grid_cartesian=normal_opts['normal_beta_sigma_grid_cartesian'],
                    use_newton=normal_opts['normal_map_use_newton'],
                )
                # Freeze BLUP beta after first MAP step so subsequent MAP warm-starts
                # (which drift further from OLS) don't degrade BLUP accuracy.
                if _outer_i == 0 and n_outer > 1 and 'normal_map_beta_for_blup' in stats:
                    _blup_beta_anchor = stats['normal_map_beta_for_blup']
            if map_refine and normal_opts['normal_laplace_eb'] and _priors_ok:
                if _is_last and _blup_beta_anchor is not None:
                    stats = dict(stats)
                    stats['normal_map_beta_for_blup'] = _blup_beta_anchor
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
                    n_steps=normal_opts['normal_laplace_eb_steps'],
                    moment_blend=normal_opts['normal_laplace_eb_moment_blend'],
                    prior_weight=normal_opts['normal_laplace_eb_prior_weight'],
                    recompute_blup=normal_opts['normal_laplace_eb_recompute_blup'],
                    beta_alpha_low=normal_opts['beta_alpha_low'],
                    beta_alpha_high=normal_opts['beta_alpha_high'],
                    sigma_grid_refine=normal_opts['normal_laplace_eb_sigma_grid_refine'],
                    sigma_grid_scales=normal_opts['normal_laplace_eb_sigma_grid_scales'],
                    beta_tail_grid=normal_opts['normal_beta_tail_grid'] and _is_last,
                    beta_tail_grid_scales=normal_opts['normal_beta_tail_grid_scales'],
                    beta_tail_grid_min_d=normal_opts['normal_beta_tail_grid_min_d'],
                    beta_tail_grid_min_cond=normal_opts['normal_beta_tail_grid_min_cond'],
                    beta_tail_grid_blend=normal_opts['normal_beta_tail_grid_blend'],
                    beta_tail_grid_unconditional=normal_opts['normal_beta_tail_grid_unconditional'],
                    beta_tail_grid_cartesian=normal_opts['normal_beta_tail_grid_cartesian'],
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
        if map_refine and beb['bernoulli_laplace_eb_mode'] != 'off' and Zm.shape[-1] > 0:
            if beb['bernoulli_laplace_eb_mode'] == 'all':
                gate = torch.ones(Xm.shape[0], device=Xm.device, dtype=torch.bool)
                stats = _callBernoulliLaplaceEb(
                    stats, Xm, ym, Zm, mask_n, mask_m, map_priors, mask_d, mask_q, beb
                )
            else:
                gate = _bernoulliLaplaceEbGate(
                    stats,
                    Xm,
                    Zm,
                    mask_n,
                    mask_q,
                    mask_d,
                    beb['bernoulli_laplace_eb_gate_min_d'],
                    beb['bernoulli_laplace_eb_gate_min_sigma'],
                    beb['bernoulli_laplace_eb_gate_eta_abs'],
                )
                if gate.any():
                    sliced_priors = {k: _sliceBatch(v, gate) for k, v in map_priors.items()}
                    refined = _callBernoulliLaplaceEb(
                        _sliceStatsBatch(stats, gate, Xm.shape[0]),
                        Xm[gate],
                        ym[gate],
                        Zm[gate],
                        mask_n[gate],
                        mask_m[gate],
                        sliced_priors,
                        _sliceBatch(mask_d, gate),
                        _sliceBatch(mask_q, gate),
                        beb,
                    )
                    stats = _mergeStatsBatch(stats, refined, gate, Xm.shape[0])
            if beb['bernoulli_laplace_eb_diagnostics']:
                _addLaplaceEbSkippedDiagnostics(stats, gate, Xm.dtype)
    elif likelihood_family == 2:
        stats = lmmPoisson(Xm, ym, Zm, mask_n, mask_m, ns, n_total, uncorr=uncorr, **kwargs)
        stats = refinePoissonPath(
            stats,
            Xm,
            ym,
            Zm,
            mask_n,
            mask_m,
            map_priors,
            mask_d=mask_d,
            mask_q=mask_q,
            map_refine=map_refine,
            options=poisson_refinement_options,
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
    'refinePoissonLaplacePirlsSigmaAverage',
    'refinePoissonMarginalMeanBeta',
    'refinePoissonVariationalGaussian',
    'refinePoissonVariationalGaussianSigmaAverage',
]
