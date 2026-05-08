"""Public dispatchers for analytical LMM/GLMM variance-component estimators."""

import torch

from metabeta.analytical.blup import analyticalBLUPContext
from metabeta.analytical.normal import lmmNormal
from metabeta.analytical.pql import lmmBernoulli, lmmPoisson


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
    uncorr = (eta_rfx == 0) if eta_rfx is not None else None  # (B,) bool or None
    if likelihood_family == 0:
        stats = lmmNormal(Xm, ym, Zm, mask_n, mask_m, ns, n_total, uncorr=uncorr, mask_q=mask_q)
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
]
