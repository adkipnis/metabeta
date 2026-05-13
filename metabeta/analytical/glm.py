"""Batched GLM estimators used to initialize analytical GLMMs."""

import torch

from metabeta.analytical.linalg import _adaptiveRidge, _safeSolve
from metabeta.analytical.working import _poissonMeanDerivative


def olsNormal(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled OLS via normal equations (X'X + ridge)^{-1} X'y, plus residual SD."""
    d = Xm.shape[-1]
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    beta = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)

    yhat = torch.einsum('bmnd,bd->bmn', Xm, beta)
    resid = ym - yhat * mask
    ss_resid = resid.square().sum(dim=(1, 2))
    df = (n_total - d).clamp(min=1)
    sigma_eps_ols = (ss_resid / df).sqrt().unsqueeze(-1)
    return beta, sigma_eps_ols


def irlsBernoulli(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 8,
    damping: float = 0.7,
    clamp: float = 20.0,
    nu_ffx: torch.Tensor | None = None,
    tau_ffx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched IRLS logistic regression via compact normal equations.

    When nu_ffx and tau_ffx are provided, applies a Gaussian prior N(nu_ffx, diag(tau_ffx²))
    by adding diag(1/τ²) to XwX and diag(1/τ²)ν to Xwz at each step.
    """
    B, d = Xm.shape[0], Xm.shape[-1]
    beta = Xm.new_zeros(B, d)
    prior_prec = (
        torch.where(
            tau_ffx > 0, 1.0 / tau_ffx.clamp(min=1e-4).square(), tau_ffx.new_zeros(tau_ffx.shape)
        )
        if (nu_ffx is not None and tau_ffx is not None)
        else None
    )
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        if prior_prec is not None:
            XwX = XwX + torch.diag_embed(prior_prec)
            Xwz = Xwz + prior_prec * nu_ffx
        beta_new = _safeSolve(XwX + _adaptiveRidge(XwX), Xwz)
        beta = (damping * beta_new + (1.0 - damping) * beta).nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0
        )
        beta = beta.clamp(-clamp, clamp)
    return beta


def irlsPoisson(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
    damping: float = 0.5,
) -> torch.Tensor:
    """Batched IRLS Poisson regression via compact normal equations."""
    B, d = Xm.shape[0], Xm.shape[-1]
    n_obs = mask.sum(dim=(1, 2)).clamp(min=1)
    y_mean = (ym.sum(dim=(1, 2)) / n_obs).clamp(min=1e-4)
    beta = Xm.new_zeros(B, d)
    beta[:, 0] = torch.log(y_mean)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        mu, deriv = _poissonMeanDerivative(eta)
        grad_mu = (mu * deriv.clamp(min=1e-6)).clamp(min=1e-12)
        w = mu * deriv.square() * mask
        z = (eta + (ym - mu * mask) / grad_mu) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta_new = _safeSolve(XwX + _adaptiveRidge(XwX), Xwz)
        beta = damping * beta_new + (1 - damping) * beta
    return beta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
