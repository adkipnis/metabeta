"""Public dispatchers for analytical LMM/GLMM variance-component estimators."""

import torch

from metabeta.analytical.blup import analyticalBLUPContext
from metabeta.analytical.map import refineNormalMapSrfx
from metabeta.analytical.normal import lmmNormal
from metabeta.analytical.pql import lmmBernoulli, lmmPoisson

_MAP_PRIOR_KEYS = (
    'nu_ffx',
    'tau_ffx',
    'family_ffx',
    'tau_rfx',
    'family_sigma_rfx',
    'tau_eps',
    'family_sigma_eps',
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
    map_lr = kwargs.pop('map_lr', 0.03)
    map_recompute_blup = kwargs.pop('map_recompute_blup', True)
    map_optimize = kwargs.pop('map_optimize', 'all')
    mask_d = kwargs.pop('mask_d', None)
    uncorr = (eta_rfx == 0) if eta_rfx is not None else None  # (B,) bool or None
    if likelihood_family == 0:
        stats = lmmNormal(Xm, ym, Zm, mask_n, mask_m, ns, n_total, uncorr=uncorr, mask_q=mask_q)
        if map_refine and Zm.shape[-1] > 0 and all(v is not None for v in map_priors.values()):
            stats = refineNormalMapSrfx(
                stats,
                Xm,
                ym,
                Zm,
                mask_n,
                mask_m,
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
                lr=map_lr,
                recompute_blup=map_recompute_blup,
                optimize=map_optimize,
            )
    elif likelihood_family == 1:
        stats = lmmBernoulli(Xm, ym, Zm, mask_n, mask_m, ns, n_total, uncorr=uncorr, **kwargs)
    elif likelihood_family == 2:
        stats = lmmPoisson(Xm, ym, Zm, mask_n, mask_m, ns, n_total, uncorr=uncorr, **kwargs)
    else:
        raise ValueError(f'unsupported likelihood_family={likelihood_family}')

    return stats


__all__ = [
    'analyticalBLUPContext',
    'glmm',
    'lmmBernoulli',
    'lmmNormal',
    'lmmPoisson',
    'refineNormalMapSrfx',
]
