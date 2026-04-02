"""Batched OLS and IRLS solvers in two formulations.

flat      — pass the tall (B, m*n, d) matrix directly to lstsq; better
            conditioned because the normal equations are never formed.
compacted — form XtX / XwX explicitly (B, d, d) before solving; cheaper
            when m*n >> d but squares the condition number.

The flat variants are the defaults used in production.
"""

import torch


# ---------------------------------------------------------------------------
# OLS (Normal likelihood)
# ---------------------------------------------------------------------------


def olsNormal(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_total: torch.Tensor,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled OLS via tall-matrix lstsq, plus residual SD."""
    B, m, n, d = Xm.shape
    beta = torch.linalg.lstsq(
        Xm.reshape(B, m * n, d),
        ym.reshape(B, m * n, 1),
    ).solution.squeeze(-1)

    yhat = torch.einsum('bmnd,bd->bmn', X, beta)
    resid = ym - yhat * mask
    ss_resid = resid.square().sum(dim=(1, 2))
    df = (n_total - d).clamp(min=1)
    sigma_eps_ols = (ss_resid / df).sqrt().unsqueeze(-1)
    return beta, sigma_eps_ols


def olsNormalCompacted(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_total: torch.Tensor,
    X: torch.Tensor,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled OLS via normal equations (X'X + εI)^{-1} X'y, plus residual SD."""
    d = Xm.shape[-1]
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    eye = ridge * torch.eye(d, device=Xm.device, dtype=Xm.dtype).expand_as(XtX)
    beta = torch.linalg.solve(XtX + eye, Xty)

    yhat = torch.einsum('bmnd,bd->bmn', X, beta)
    resid = ym - yhat * mask
    ss_resid = resid.square().sum(dim=(1, 2))
    df = (n_total - d).clamp(min=1)
    sigma_eps_ols = (ss_resid / df).sqrt().unsqueeze(-1)
    return beta, sigma_eps_ols


# ---------------------------------------------------------------------------
# IRLS (Bernoulli / logistic)
# ---------------------------------------------------------------------------


def irlsBernoulli(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
) -> torch.Tensor:
    """Batched IRLS logistic regression via tall-matrix WLS."""
    B, m, n, d = Xm.shape
    beta = Xm.new_zeros(B, d)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        w_sqrt = w.sqrt()
        Xw = (Xm * w_sqrt.unsqueeze(-1)).reshape(B, m * n, d)
        zw = (z * w_sqrt).reshape(B, m * n, 1)
        beta = torch.linalg.lstsq(Xw, zw).solution.squeeze(-1)
    return beta


def irlsBernoulliCompacted(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Batched IRLS logistic regression via normal equations XwX."""
    d = Xm.shape[-1]
    beta = Xm.new_zeros(Xm.shape[0], d)
    eye = ridge * torch.eye(d, device=Xm.device, dtype=Xm.dtype)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta = torch.linalg.solve(XwX + eye, Xwz)
    return beta


# ---------------------------------------------------------------------------
# IRLS (Poisson)
# ---------------------------------------------------------------------------


def irlsPoisson(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
    damping: float = 0.5,
) -> torch.Tensor:
    """Batched IRLS Poisson regression via tall-matrix WLS (damped)."""
    B, m, n, d = Xm.shape
    n_obs = mask.sum(dim=(1, 2)).clamp(min=1)
    y_mean = (ym.sum(dim=(1, 2)) / n_obs).clamp(min=1e-4)
    beta = Xm.new_zeros(B, d)
    beta[:, 0] = torch.log(y_mean)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        mu = torch.exp(eta.clamp(max=20))
        w = (mu * mask).clamp(min=1e-6)
        z = (eta + (ym - mu * mask) / w) * mask
        w_sqrt = w.sqrt()
        Xw = (Xm * w_sqrt.unsqueeze(-1)).reshape(B, m * n, d)
        zw = (z * w_sqrt).reshape(B, m * n, 1)
        beta_new = torch.linalg.lstsq(Xw, zw).solution.squeeze(-1)
        beta = damping * beta_new + (1 - damping) * beta
    return beta


def irlsPoissonCompacted(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
    damping: float = 0.5,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Batched IRLS Poisson regression via normal equations XwX (damped)."""
    B, m, n, d = Xm.shape
    n_obs = mask.sum(dim=(1, 2)).clamp(min=1)
    y_mean = (ym.sum(dim=(1, 2)) / n_obs).clamp(min=1e-4)
    beta = Xm.new_zeros(B, d)
    beta[:, 0] = torch.log(y_mean)
    eye = ridge * torch.eye(d, device=Xm.device, dtype=Xm.dtype)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        mu = torch.exp(eta.clamp(max=20))
        w = (mu * mask).clamp(min=1e-6)
        z = (eta + (ym - mu * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta_new = torch.linalg.solve(XwX + eye, Xwz)
        beta = damping * beta_new + (1 - damping) * beta
    return beta
