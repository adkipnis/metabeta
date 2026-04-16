"""Batched GLM (Generalized Linear Model) estimators in two formulations.

GLM = no random effects; regression on pooled or grouped data.  Pairs with
lmm.py, which adds group-level random effects.

flat      — pass the tall (B, m*n, d) matrix directly to lstsq; better
            conditioned because the normal equations are never formed.
            NOTE: CUDA's lstsq driver (gels) requires full column rank and
            will error when X has zero-padded columns (variable-d batches).
            Use the compacted variants in production.

compacted — form XtX / XwX explicitly (B, d, d) before solving; cheaper
            when m*n >> d. Uses a scale-adaptive ridge εI scaled by the
            largest diagonal entry, so exactly-zero columns (from variable-d
            padding) produce beta_j = 0 without perturbing active columns.
"""

import torch


def _safeSolve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.linalg.solve with a per-element ridge boost for singular batches.

    torch.linalg.solve raises LinAlgError if *any* batch element is singular,
    aborting the whole batch. On failure this function:
      1. Computes the minimum eigenvalue per batch element (eigvalsh — valid
         because all A in glmm.py are symmetric by construction).
      2. Adds just enough ridge to bring each element's condition number to ≤1e4;
         well-conditioned elements receive zero additional boost.
      3. Retries; falls back to zeros only if that also fails (degenerate edge).
    """
    try:
        return torch.linalg.solve(A, b)
    except torch.linalg.LinAlgError:
        eigvals = torch.linalg.eigvalsh(A)           # (..., n), ascending
        max_eig = eigvals[..., -1].clamp(min=1.0)    # (...)
        boost = (max_eig * 1e-4 - eigvals[..., 0]).clamp(min=0.0)  # 0 for well-conditioned
        d = A.shape[-1]
        eye = torch.eye(d, device=A.device, dtype=A.dtype)
        A_fixed = A + boost[..., None, None] * eye
        try:
            return torch.linalg.solve(A_fixed, b)
        except torch.linalg.LinAlgError:
            return torch.zeros_like(b)


def _adaptiveRidge(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-adaptive ridge: eps * max_diag(A) * I.

    For active columns the ridge is eps-fraction of their scale (negligible).
    For zero-padded columns (diagonal exactly 0) the ridge is eps * max_active,
    which is large enough to be nonsingular while keeping beta_j ≈ 0 because
    the corresponding rhs entry is also zero.
    """
    d = A.shape[-1]
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)  # (B,)
    return eps * scale.view(-1, 1, 1) * torch.eye(d, device=A.device, dtype=A.dtype)


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
    """Pooled OLS via tall-matrix lstsq, plus residual SD.

    NOTE: fails on CUDA when X has zero-padded columns (variable-d batches).
    """
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled OLS via normal equations (X'X + ridge)^{-1} X'y, plus residual SD."""
    d = Xm.shape[-1]
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    beta = torch.linalg.solve(XtX + _adaptiveRidge(XtX), Xty)

    yhat = torch.einsum('bmnd,bd->bmn', Xm, beta)
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
    """Batched IRLS logistic regression via tall-matrix WLS.

    NOTE: fails on CUDA when X has zero-padded columns (variable-d batches).
    """
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
) -> torch.Tensor:
    """Batched IRLS logistic regression via normal equations XwX."""
    B, d = Xm.shape[0], Xm.shape[-1]
    beta = Xm.new_zeros(B, d)
    for _ in range(n_iter):
        eta = torch.einsum('bmnd,bd->bmn', Xm, beta)
        p = torch.sigmoid(eta)
        w = (p * (1 - p) * mask).clamp(min=1e-6)
        z = (eta + (ym - p * mask) / w) * mask
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta = torch.linalg.solve(XwX + _adaptiveRidge(XwX), Xwz)
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
    """Batched IRLS Poisson regression via tall-matrix WLS (damped).

    NOTE: fails on CUDA when X has zero-padded columns (variable-d batches).
    """
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
) -> torch.Tensor:
    """Batched IRLS Poisson regression via normal equations XwX (damped)."""
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
        XwX = torch.einsum('bmnd,bmn,bmnk->bdk', Xm, w, Xm)
        Xwz = torch.einsum('bmnd,bmn->bd', Xm, w * z)
        beta_new = torch.linalg.solve(XwX + _adaptiveRidge(XwX), Xwz)
        beta = damping * beta_new + (1 - damping) * beta
    return beta
