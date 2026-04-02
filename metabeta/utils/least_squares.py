import torch


def olsNormal(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_total: torch.Tensor,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled OLS: beta = (X'X)^{-1} X'y, plus residual SD."""
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    beta = torch.linalg.lstsq(XtX, Xty).solution

    yhat = torch.einsum('bmnd,bd->bmn', X, beta)
    resid = ym - yhat * mask
    ss_resid = resid.square().sum(dim=(1, 2))
    df = (n_total - X.shape[-1]).clamp(min=1)
    sigma_eps_ols = (ss_resid / df).sqrt().unsqueeze(-1)
    return beta, sigma_eps_ols


def irlsBernoulli(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
) -> torch.Tensor:
    """Batched IRLS logistic regression (fixed iterations)."""
    B, _, _, d = Xm.shape
    beta = Xm.new_zeros(B, d)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta = torch.linalg.lstsq(XwX, Xwz).solution
    return beta


def irlsPoisson(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 5,
    damping: float = 0.5,
) -> torch.Tensor:
    """Batched IRLS Poisson regression (damped, warm-started)."""
    B, _, _, d = Xm.shape
    # warm-start: intercept = log(mean(y)), slopes = 0
    n_obs = mask.sum(dim=(1, 2)).clamp(min=1)
    y_mean = (ym.sum(dim=(1, 2)) / n_obs).clamp(min=1e-4)
    beta = Xm.new_zeros(B, d)
    beta[:, 0] = torch.log(y_mean)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        mu = torch.exp(eta.clamp(max=20))
        w = (mu * mask).clamp(min=1e-6)
        z = (eta + (ym - mu * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta_new = torch.linalg.lstsq(XwX, Xwz).solution
        beta = damping * beta_new + (1 - damping) * beta
    return beta
