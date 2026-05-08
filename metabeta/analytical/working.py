"""Family-specific working quantities for GLMM estimators."""

import torch
import torch.nn.functional as F

from metabeta.analytical.constants import (
    _POISSON_ETA_CLIP_MAX,
    _POISSON_ETA_TAPER_WIDTH,
)


def _poissonMeanDerivative(eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Poisson mean and d min(eta, cap) / d eta with a short taper near the cap."""
    eta_eff = eta.clamp(max=_POISSON_ETA_CLIP_MAX)
    mu = torch.exp(eta_eff)
    taper_start = _POISSON_ETA_CLIP_MAX - _POISSON_ETA_TAPER_WIDTH
    deriv = ((_POISSON_ETA_CLIP_MAX - eta) / _POISSON_ETA_TAPER_WIDTH).clamp(0.0, 1.0)
    deriv = torch.where(eta <= taper_start, torch.ones_like(deriv), deriv)
    return mu, deriv


def _pqlWorking(
    eta: torch.Tensor,
    y: torch.Tensor,
    mask_n: torch.Tensor,
    likelihood_family: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PQL working weights w and working response ỹ at linearisation point η.

    Bernoulli/logit : w = μ(1-μ),  ỹ = η + (y-μ)/w
    Poisson/log     : w = μ,        ỹ = η + (y-μ)/w
    Inactive observations are zeroed via mask_n.
    """
    if likelihood_family == 1:
        mu = torch.sigmoid(eta)
        w = (mu * (1.0 - mu)).clamp(min=1e-6) * mask_n
        ytilde = (eta + (y - mu) / w.clamp(min=1e-30)) * mask_n
    elif likelihood_family == 2:
        mu, deriv = _poissonMeanDerivative(eta)
        grad_mu = (mu * deriv.clamp(min=1e-6)).clamp(min=1e-12)
        w = mu * deriv.square() * mask_n
        ytilde = (eta + (y - mu) / grad_mu) * mask_n
    else:
        raise ValueError(f'unsupported likelihood_family={likelihood_family}')
    return w, ytilde


def _groupNll(
    bg: torch.Tensor,
    beta_0: torch.Tensor,
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    Psi_inv: torch.Tensor,
    likelihood_family: int,
) -> torch.Tensor:
    """Per-group negative log-posterior  f(bᵍ) = −Σᵢ log p(yᵢ|ηᵢ) + ½ bᵍᵀΨ⁻¹bᵍ.

    Returns shape (B, m).  Used by the Armijo line-search inside the Newton loop.
    """
    eta = torch.einsum('bmnd,bd->bmn', Xm, beta_0) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    if likelihood_family == 1:
        nll_obs = (-ym * F.logsigmoid(eta) - (1.0 - ym) * F.logsigmoid(-eta)) * mask_n
    else:
        eta_eff = eta.clamp(max=_POISSON_ETA_CLIP_MAX)
        mu = torch.exp(eta_eff)
        nll_obs = (-(ym * eta_eff - mu)) * mask_n
    return nll_obs.sum(dim=-1) + 0.5 * torch.einsum('bmq,bqr,bmr->bm', bg, Psi_inv, bg)
