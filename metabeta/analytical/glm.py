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
    family_ffx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched IRLS logistic regression via compact normal equations.

    When nu_ffx and tau_ffx are provided, applies a MAP prior at each IRLS step:
    - Normal (family_ffx == 0 or family_ffx is None): Gaussian N(nu_ffx, diag(tau_ffx²));
      constant precision 1/τ² added to XwX and diag(1/τ²)ν to Xwz.
    - Student-t (family_ffx == 1, df=5): precision (df+1)/(df·τ²+(β−ν)²) is recomputed
      from the current beta at each iteration (EM-style adaptive ridge).
    Inactive dimensions (tau_ffx == 0) always receive zero prior precision.
    """
    B, d = Xm.shape[0], Xm.shape[-1]
    beta = Xm.new_zeros(B, d)
    if nu_ffx is not None and tau_ffx is not None:
        active_mask = tau_ffx > 0  # (B, d)
        zeros = tau_ffx.new_zeros(tau_ffx.shape)
        normal_prec = torch.where(active_mask, 1.0 / tau_ffx.clamp(min=1e-4).square(), zeros)
        # (B, 1) bool — True for datasets with Student-t FFX prior (df=5)
        is_student = (family_ffx == 1).unsqueeze(-1) if family_ffx is not None else None
    else:
        normal_prec = None
        is_student = None
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        if normal_prec is not None:
            if is_student is not None:
                # Adaptive Student-t precision: (df+1)/(df·τ²+(β−ν)²), df=5
                student_prec = torch.where(
                    active_mask,
                    6.0
                    / (5.0 * tau_ffx.clamp(min=1e-8).square() + (beta - nu_ffx).square()).clamp(
                        min=1e-8
                    ),
                    zeros,
                )
                prior_prec = torch.where(is_student, student_prec, normal_prec)
            else:
                prior_prec = normal_prec
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
