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


def _expandBatchMask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand a batch-shaped mask across the matrix/vector solve dimensions."""
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(-1)
    return mask


def _safeSolve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.linalg.solve with escalating ridge fallback for singular batches.

    torch.linalg.solve raises LinAlgError if *any* batch element is singular,
    aborting the whole batch. On failure this function:
      1. Uses solve_ex to recover any batch elements that still solve cleanly.
      2. Adds escalating diagonal jitter to the failed elements without relying
         on an eigendecomposition, which can fail for ill-conditioned batches.
      3. Falls back to zeros only for elements that remain degenerate.
    """
    try:
        solution = torch.linalg.solve(A, b)
        if torch.isfinite(solution).all():
            return solution
    except torch.linalg.LinAlgError:
        pass

    A_safe = A.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    b_safe = b.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    try:
        solution, info = torch.linalg.solve_ex(A_safe, b_safe, check_errors=False)
    except torch.linalg.LinAlgError:
        solution = torch.zeros_like(b_safe)
        info = torch.ones(A_safe.shape[:-2], device=A.device, dtype=torch.int32)

    solution = solution.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    ok = info == 0
    finite = torch.isfinite(solution).flatten(start_dim=ok.dim()).all(dim=-1)
    ok = ok & finite

    d = A.shape[-1]
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    scale = A_safe.abs().flatten(start_dim=-2).amax(dim=-1).clamp(min=1.0)
    for eps in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
        if bool(ok.all()):
            break
        A_fixed = A_safe + (eps * scale)[..., None, None] * eye
        try:
            candidate, candidate_info = torch.linalg.solve_ex(A_fixed, b_safe, check_errors=False)
        except torch.linalg.LinAlgError:
            continue
        candidate = candidate.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        candidate_ok = candidate_info == 0
        candidate_finite = (
            torch.isfinite(candidate).flatten(start_dim=candidate_ok.dim()).all(dim=-1)
        )
        candidate_ok = candidate_ok & candidate_finite
        take = candidate_ok & ~ok
        solution = torch.where(_expandBatchMask(take, solution), candidate, solution)
        ok = ok | take

    return torch.where(_expandBatchMask(ok, solution), solution, torch.zeros_like(solution))


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
    n_iter: int = 8,
    damping: float = 0.7,
    clamp: float = 20.0,
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
        beta_new = _safeSolve(XwX + _adaptiveRidge(XwX), Xwz)
        beta = (damping * beta_new + (1.0 - damping) * beta).nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0
        )
        beta = beta.clamp(-clamp, clamp)
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
