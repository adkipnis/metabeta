"""Linear algebra helpers for analytical GLMM estimators."""

import torch

from metabeta.analytical.constants import (
    _EIGH_JITTERS,
    _NORMAL_RANK_ABS_TOL,
    _NORMAL_RANK_REL_TOL,
    _NORMAL_Z_COND_CAP,
)


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
    """Scale-adaptive ridge for batched square systems."""
    d = A.shape[-1]
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)
    return eps * scale[..., None, None] * torch.eye(d, device=A.device, dtype=A.dtype)


def _adaptiveRidgeBm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-adaptive ridge for (B, m, q, q) block matrices."""
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)  # (B, m)
    q = A.shape[-1]
    return eps * scale[..., None, None] * torch.eye(q, device=A.device, dtype=A.dtype)


def _rankFromEigenvalues(
    vals: torch.Tensor,
    rel_tol: float = _NORMAL_RANK_REL_TOL,
    abs_tol: float = _NORMAL_RANK_ABS_TOL,
) -> torch.Tensor:
    """Numerical rank from non-negative eigenvalues."""
    vals = vals.clamp(min=0.0)
    scale = vals.amax(dim=-1, keepdim=True)
    tol = torch.maximum(scale * rel_tol, vals.new_tensor(abs_tol))
    return (vals > tol).sum(dim=-1)


def _groupZDiagnostics(
    ZtZ: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    q: int,
    cond_cap: float = _NORMAL_Z_COND_CAP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify groups whose random-effect design can estimate all q components."""
    B, m, _, _ = ZtZ.shape
    vals, _ = _eighWithJitter(ZtZ.reshape(B * m, q, q))
    vals = vals.clamp(min=0.0).reshape(B, m, q)
    ranks = _rankFromEigenvalues(vals).to(ZtZ.dtype)

    max_eig = vals.amax(dim=-1)
    rank_tol = torch.maximum(
        max_eig[..., None] * _NORMAL_RANK_REL_TOL,
        vals.new_tensor(_NORMAL_RANK_ABS_TOL),
    )
    min_active_eig = torch.where(vals > rank_tol, vals, vals.new_full((), float('inf'))).amin(
        dim=-1
    )
    cond = max_eig / min_active_eig.clamp(min=1e-30)

    diag_sum = (ZtZ.diagonal(dim1=-2, dim2=-1).clamp(min=0.0) * mask_m[:, :, None]).sum(dim=1)
    diag_scale = diag_sum.amax(dim=-1, keepdim=True)
    diag_tol = torch.maximum(
        diag_scale * _NORMAL_RANK_REL_TOL,
        diag_sum.new_tensor(_NORMAL_RANK_ABS_TOL),
    )
    active_components = diag_sum > diag_tol
    active_count = active_components.sum(dim=-1).to(ZtZ.dtype)
    informative = (
        mask_m.bool()
        & (active_count[:, None] > 0)
        & (ns > active_count[:, None] + 1)
        & (ranks >= active_count[:, None])
        & (cond <= cond_cap)
    )
    return informative.to(ZtZ.dtype), ranks * mask_m, cond, active_components, active_count


def _gramRank(A: torch.Tensor) -> torch.Tensor:
    """Numerical rank of a batched Gram matrix."""
    vals, _ = _eighWithJitter(0.5 * (A + A.mT))
    return _rankFromEigenvalues(vals).to(A.dtype)


def _maskedMean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    count = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / count


def _maskedMedian(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = torch.where(mask.bool(), values, values.new_full((), float('inf')))
    sorted_vals = masked.sort(dim=dim).values
    count = mask.sum(dim=dim).long()
    kth = ((count - 1).clamp(min=0)).unsqueeze(dim)
    kth = kth.expand(*sorted_vals.shape[:dim], 1, *sorted_vals.shape[dim + 1 :])
    median = sorted_vals.gather(dim, kth).squeeze(dim)
    return torch.where(count > 0, median, torch.zeros_like(median))


def _eighWithJitter(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch eigh with fallback to element-wise jitter for ill-conditioned matrices.

    Fast path: full-batch eigh with no overhead.
    On failure: retry each element individually with increasing diagonal jitter.
    Elements that fail even at max jitter get vals=0, vecs=I (soft failure).
    """
    try:
        return torch.linalg.eigh(M)
    except torch.linalg.LinAlgError:
        pass

    B, q, _ = M.shape
    eye = torch.eye(q, device=M.device, dtype=M.dtype)
    all_vals, all_vecs = [], []
    for i in range(B):
        Mi = M[i]
        result = None
        for jitter in _EIGH_JITTERS:
            try:
                result = torch.linalg.eigh(Mi + jitter * eye)
                break
            except torch.linalg.LinAlgError:
                continue
        if result is None:
            result = (torch.zeros(q, device=M.device, dtype=M.dtype), eye)
        all_vals.append(result[0])
        all_vecs.append(result[1])
    return torch.stack(all_vals), torch.stack(all_vecs)


def _psdProject(M: torch.Tensor) -> torch.Tensor:
    """Project a symmetric (B, q, q) matrix onto the PSD cone."""
    M = 0.5 * (M + M.mT)
    vals, vecs = _eighWithJitter(M)
    return vecs @ torch.diag_embed(vals.clamp(min=0.0)) @ vecs.mT


def _psdClampEigenvalues(M: torch.Tensor, max_eig: torch.Tensor | float) -> torch.Tensor:
    """Project to PSD and cap eigenvalues for arbitrary leading batch dimensions."""
    M = 0.5 * (M + M.mT)
    leading_shape = M.shape[:-2]
    q = M.shape[-1]
    flat_M = M.reshape(-1, q, q)
    vals, vecs = _eighWithJitter(flat_M)
    vals = vals.clamp(min=0.0)

    if isinstance(max_eig, torch.Tensor):
        cap = max_eig.to(device=M.device, dtype=M.dtype).reshape(-1, 1)
    else:
        cap = M.new_full((flat_M.shape[0], 1), float(max_eig))
    vals = torch.minimum(vals, cap.clamp(min=0.0))
    return (vecs @ torch.diag_embed(vals) @ vecs.mT).reshape(*leading_shape, q, q)


def _shrinkOffDiagonal(M: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Shrink covariance off-diagonals toward zero while preserving the diagonal."""
    diag = torch.diag_embed(M.diagonal(dim1=-2, dim2=-1))
    return alpha[:, None, None] * M + (1.0 - alpha[:, None, None]) * diag


def _pseudoInverse(M: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse of a PSD (B, q, q) matrix via eigendecomposition."""
    vals, vecs = _eighWithJitter(M)
    tol = vals.amax(dim=-1, keepdim=True).clamp(min=1e-8) * 1e-6
    inv_vals = torch.where(vals > tol, 1.0 / vals.clamp(min=1e-30), torch.zeros_like(vals))
    return vecs @ torch.diag_embed(inv_vals) @ vecs.mT


def _ridgeInv(M: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """Ridge-regularized inverse of a PSD (B, q, q) matrix.

    floor: (B,) per-batch scalar added to all eigenvalues before inversion.
    Unlike pseudoinverse, never zeros any eigenvalue — ensures strong shrinkage
    when M eigenvalues are near or below floor.
    """
    vals, vecs = _eighWithJitter(M)
    inv_vals = 1.0 / (vals + floor[:, None]).clamp(min=1e-30)
    return vecs @ torch.diag_embed(inv_vals) @ vecs.mT
