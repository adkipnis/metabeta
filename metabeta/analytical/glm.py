"""Batched compact GLM estimators used to initialize analytical GLMMs."""

import torch

from metabeta.analytical.linalg import _adaptiveRidge, _safeSolve
from metabeta.analytical.working import _poissonMeanDerivative


def olsNormalCompacted(
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


def irlsBernoulliCompacted(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 8,
    damping: float = 0.7,
    clamp: float = 20.0,
) -> torch.Tensor:
    """Batched IRLS logistic regression via compact normal equations."""
    B, d = Xm.shape[0], Xm.shape[-1]
    beta = Xm.new_zeros(B, d)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta_new = _safeSolve(XwX + _adaptiveRidge(XwX), Xwz)
        beta = (damping * beta_new + (1.0 - damping) * beta).nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0
        )
        beta = beta.clamp(-clamp, clamp)
    return beta


def irlsPoissonCompacted(
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
