"""LMM/GLMM variance-component estimators: lmmNormal (Gaussian),
lmmBernoulli (logit), lmmPoisson (log). All return the same dict keys.
"""

import torch
import torch.nn.functional as F

from metabeta.utils.glm import (
    _adaptiveRidge,
    _safeSolve,
    irlsBernoulliCompacted,
    irlsPoissonCompacted,
)
from metabeta.utils.regularization import unconstrainedToCholesky


def _adaptiveRidgeBm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-adaptive ridge for (B, m, q, q) block matrices."""
    scale = A.diagonal(dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1.0)  # (B, m)
    q = A.shape[-1]
    return eps * scale[..., None, None] * torch.eye(q, device=A.device, dtype=A.dtype)


_EIGH_JITTERS = [1e-6, 1e-4, 1e-2, 1.0]
_BERNOULLI_INITIAL_PSI_FLOOR = 0.25
_BERNOULLI_HG_INV_EIG_CAP = 25.0
_BERNOULLI_PSI_EIG_CAP = 49.0
_BERNOULLI_BLUP_VAR_INFLATION = 1.5
_BERNOULLI_CORR_SHRINKAGE_C = 10.0
_BERNOULLI_BLUP_KH_VAR_CAP = 0.2
_BERNOULLI_HIGH_D_BLUP_VAR_FLOOR = 0.25
_POISSON_ETA_CLIP_MAX = 10.0
_POISSON_INITIAL_PSI_FLOOR = 0.01
_POISSON_HG_INV_EIG_CAP = 25.0
_POISSON_PSI_EIG_CAP = 25.0
_POISSON_CORR_SHRINKAGE_C = 20.0
_POISSON_ETA_TAPER_WIDTH = 0.5
_POISSON_BETA_CLAMP = 10.0
_POISSON_BLUP_CLAMP = 10.0
_NORMAL_Z_COND_CAP = 1e6
_NORMAL_RANK_REL_TOL = 1e-5
_NORMAL_RANK_ABS_TOL = 1e-8
_NORMAL_FULL_MIN_EM = 5


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


def _pqlPass(
    beta: torch.Tensor,  # (B, d) fixed effects for this pass
    Psi_inv: torch.Tensor,  # (B, q, q) precision used for Newton penalty and Hg
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)
    mask_m: torch.Tensor,  # (B, m)
    mask4: torch.Tensor,  # (B, m, 1, 1)
    active: torch.Tensor,  # (B, m) bool
    eye_q: torch.Tensor,  # (q, q)
    eye_q_bm: torch.Tensor,  # (B, m, q, q)
    G: torch.Tensor,  # (B,)
    likelihood_family: int,
    n_newton: int = 3,
    bg_init: torch.Tensor | None = None,  # warm start; None = cold start from zeros
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """One PQL pass: damped Newton → Ψ̂_Lap M-step → GLS β̂ and BLUPs.

    Newton starts from bg_init (or zeros if None) under (beta, Psi_inv).
    After Newton, computes Ψ̂_Lap = mean_g(b̂_g b̂_g' + H_g^{-1}), then
    β̂_GLS and BLUPs via the Woodbury/Schur-complement GLS under Ψ̂_Lap.

    bg is clamped to ±20 after each Newton step to prevent bg_outer from
    inflating Psi_lap for overdispersed Poisson or ill-conditioned datasets.

    Returns
    -------
    beta_gls   : (B, d)       GLS-corrected fixed effects
    Psi_lap    : (B, q, q)    Laplace M-step estimate of Ψ
    blups      : (B, m, q)    per-group random effects
    Kg_inv     : (B, m, q, q) GLS posterior covariance (ZWZ + Psi_lap_inv)^{-1}
    mean_Hg_inv: (B, q, q)    mean Laplace posterior covariance across groups
    resid_gls  : (B, m, n)    working residual ỹ − Xβ̂_GLS (masked)
    blup_kh_var: (B, m, q)    beta-estimation uncertainty contribution to blup_var
    """
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]

    # --- Newton loop ---
    # Cold start from zeros (Pass 1) or warm start from previous blups (Pass 2+).
    # bg is clamped at ±20 after each step: prevents bg_outer from inflating Psi_lap
    # for Poisson datasets with large counts, where 3 unconstrained Newton steps can
    # push bg to O(100) and Psi_lap to O(10000).
    bg = bg_init.clone() if bg_init is not None else ym.new_zeros(B, m, q)
    for _ in range(n_newton):
        eta_t = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
        if likelihood_family == 1:
            mu_t = torch.sigmoid(eta_t)
            score_t = ym - mu_t
            w_t = (mu_t * (1.0 - mu_t)).clamp(min=1e-6) * mask_n
        else:
            mu_t, deriv_t = _poissonMeanDerivative(eta_t)
            score_t = (ym - mu_t) * deriv_t
            w_t = mu_t * deriv_t.square() * mask_n
        ZWZ_t = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_t, Zm)         # (B, m, q, q)
        grad_g = torch.einsum('bmnq,bmn->bmq', Zm, score_t * mask_n) - torch.einsum(  # Zᵀscore
            'bqr,bmr->bmq', Psi_inv, bg
        )  # −Ψ⁻¹b
        ZWZ_t_safe = torch.where(active[:, :, None, None], ZWZ_t, eye_q)
        Hg = ZWZ_t_safe + Psi_inv[:, None]
        delta = _safeSolve(Hg + _adaptiveRidgeBm(Hg), grad_g)             # (B, m, q)
        slope = (grad_g * delta).sum(dim=-1)                               # (B, m)
        f_old = _groupNll(bg, beta, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
        alpha = torch.ones(B, m, device=Xm.device, dtype=Xm.dtype)
        for _ls in range(10):
            bg_trial = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]
            f_new = _groupNll(bg_trial, beta, Xm, ym, Zm, mask_n, Psi_inv, likelihood_family)
            accept = (f_new <= f_old - 0.1 * alpha * slope) | ~active
            if accept.all():
                break
            alpha = torch.where(accept, alpha, alpha * 0.5)
        bg = (bg + alpha[:, :, None] * delta) * mask_m[:, :, None]
        bg = bg.clamp(-20.0, 20.0)  # prevent bg_outer blow-up in Psi_lap M-step

    # --- Final working quantities at (beta, bg) ---
    eta_f = torch.einsum('bmnd,bd->bmn', Xm, beta) + torch.einsum('bmnq,bmq->bmn', Zm, bg)
    w_f, ytilde_f = _pqlWorking(eta_f, ym, mask_n, likelihood_family)
    ZWZ_f = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w_f, Zm)             # (B, m, q, q)
    ZWZ_f_safe = torch.where(active[:, :, None, None], ZWZ_f, eye_q)

    # Ψ̂_Lap M-step: Ψ = mean_g(b̂_g b̂_g' + H_g^{-1})
    Hg_f = ZWZ_f_safe + Psi_inv[:, None]
    Hg_inv = _safeSolve(Hg_f + _adaptiveRidgeBm(Hg_f), eye_q_bm) * mask4

    if likelihood_family == 1:
        ns_g = mask_n.sum(dim=-1).clamp(min=1.0)                         # (B, m)
        y_rate = (ym * mask_n).sum(dim=-1) / ns_g                         # (B, m)
        outcome_balance = (4.0 * y_rate * (1.0 - y_rate)).clamp(0.0, 1.0)  # (B, m)
        info_g = ZWZ_f.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=0.0)
        info_weight = info_g / (info_g + float(q))
        cov_weight = (outcome_balance.sqrt() * info_weight).clamp(0.0, 1.0) * mask_m
        Hg_inv = _psdClampEigenvalues(Hg_inv, _BERNOULLI_HG_INV_EIG_CAP)
        Hg_inv = Hg_inv * cov_weight[:, :, None, None]
    elif likelihood_family == 2:
        info_g = ZWZ_f.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=0.0)
        info_weight = info_g / (info_g + float(q))
        Hg_inv = _psdClampEigenvalues(Hg_inv, _POISSON_HG_INV_EIG_CAP)
        Hg_inv = Hg_inv * (info_weight * mask_m)[:, :, None, None]

    bg_outer = torch.einsum('bmq,bmr->bmqr', bg, bg)
    Psi_lap = _psdProject((bg_outer + Hg_inv).sum(dim=1) / G[:, None, None])  # (B, q, q)
    if likelihood_family == 1:
        Psi_lap = _psdClampEigenvalues(Psi_lap, _BERNOULLI_PSI_EIG_CAP)
    elif likelihood_family == 2:
        Psi_lap = _psdClampEigenvalues(Psi_lap, _POISSON_PSI_EIG_CAP)
    mean_Hg_inv = (Hg_inv * mask4).sum(dim=1) / G[:, None, None]

    # --- β̂_GLS via Woodbury/Schur complement under freshly computed Ψ̂_Lap ---
    Psi_lap_inv = _pseudoInverse(Psi_lap)
    XWX_f = torch.einsum('bmnd,bmn,bmnk->bmdk', Xm, w_f, Xm)             # (B, m, d, d)
    XWZ_f = torch.einsum('bmnd,bmn,bmnq->bmdq', Xm, w_f, Zm)             # (B, m, d, q)
    XWy_f = torch.einsum('bmnd,bmn->bmd', Xm, w_f * ytilde_f)             # (B, m, d)
    ZWy_f = torch.einsum('bmnq,bmn->bmq', Zm, w_f * ytilde_f)             # (B, m, q)

    Kg = ZWZ_f_safe + Psi_lap_inv[:, None]                                 # (B, m, q, q)
    Kg_inv = _safeSolve(Kg + _adaptiveRidgeBm(Kg), eye_q_bm) * mask4

    ZWX_f = XWZ_f.mT                                                       # (B, m, q, d)
    A_g = XWX_f - torch.einsum(  # Schur complement
        'bmdq,bmqk->bmdk', XWZ_f, torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    )
    rhs_g = XWy_f - torch.einsum(
        'bmdq,bmq->bmd', XWZ_f, torch.einsum('bmqr,bmr->bmq', Kg_inv, ZWy_f)
    )
    sum_A = (A_g * mask4).sum(dim=1)
    sum_A_reg = sum_A + _adaptiveRidge(sum_A)
    beta_gls = _safeSolve(
        sum_A_reg,
        (rhs_g * mask_m[:, :, None]).sum(dim=1),
    )                                                                       # (B, d)

    # --- BLUPs: K_g⁻¹ Zᵀ W (ỹ − X β̂_GLS) ---
    # Clamp beta_gls before residuals: near-singular GLS produces extreme values that
    # cause eta overflow in the next Newton pass or catastrophic BLUP outliers.
    beta_clamp = _POISSON_BETA_CLAMP if likelihood_family == 2 else 50.0
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-beta_clamp, beta_clamp)
    resid_gls = (ytilde_f - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    blups = (
        torch.einsum(
            'bmqr,bmr->bmq',
            Kg_inv,
            torch.einsum('bmnq,bmn->bmq', Zm, w_f * resid_gls),
        )
        * mask_m[:, :, None]
    )                                                 # (B, m, q)

    blup_clamp = _POISSON_BLUP_CLAMP if likelihood_family == 2 else 20.0
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-blup_clamp, blup_clamp)

    eye_d = torch.eye(d, device=Xm.device, dtype=Xm.dtype).expand(B, d, d)
    beta_var = _safeSolve(sum_A_reg, eye_d).diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
    K_ZWX = torch.einsum('bmqr,bmrd->bmqd', Kg_inv, ZWX_f)
    blup_kh_var = (K_ZWX.square() * beta_var[:, None, None, :]).sum(dim=-1)
    blup_kh_var = (blup_kh_var * mask_m[:, :, None]).nan_to_num(nan=0.0, posinf=0.0)
    if likelihood_family == 1:
        kh_scale = max(min((d - 8) / 8.0, 1.0), 0.0)
        blup_kh_var = blup_kh_var.clamp(max=_BERNOULLI_BLUP_KH_VAR_CAP * kh_scale)
    else:
        blup_kh_var = torch.zeros_like(blup_kh_var)

    # Return Kg_inv for blup_var: (ZWZ + Psi_lap^{-1})^{-1} is the posterior covariance
    # of b_g under the final Psi_lap estimate — consistent with the Normal path's se²·W_g.
    return beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls, blup_kh_var


# ---------------------------------------------------------------------------
# Private normal-LMM implementations
# ---------------------------------------------------------------------------


def _lmmNormalFull(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,  # (B, m)     1 for active groups
    ns: torch.Tensor,  # (B, m)     group sizes (float, ≥ 1 for active)
    n_total: torch.Tensor,  # (B,)       total active observations
    n_em: int = 3,
    uncorr: torch.Tensor | None = None,  # (B,) bool — force Ψ diagonal for these datasets
    mask_q: torch.Tensor | None = None,  # (B, q) bool — active random-effect components
) -> dict[str, torch.Tensor]:
    """GLS estimator for the LME y_g = X_g β + Z_g b_g + ε_g, b_g ~ N(0, Ψ).

    Three-stage pipeline: (1) within-Z projection for σ̂_ε, (2) MoM for Ψ̂,
    (3) Woodbury GLS for β̂ and BLUPs; followed by EM refinement of Ψ and σ_ε.
    Handles arbitrary q (including q=1).

    Returns a dict with keys: beta_est, beta_wg, sigma_eps_est, sigma_rfx_est,
    blup_est, blup_var, bhat, resid_g, Psi.
    """
    B, m, n, d = Xm.shape
    q = Zm.shape[-1]
    if mask_q is not None:
        Zm = Zm * mask_q.to(device=Zm.device, dtype=Zm.dtype)[:, None, None, :q]
    N = n_total.float()                               # (B,)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,)
    active = mask_m.bool()                            # (B, m)

    eye_q = torch.eye(q, device=Xm.device, dtype=Xm.dtype)       # (q, q)
    eye_q_bm = eye_q.expand(B, m, q, q)                          # (B, m, q, q)

    # ------------------------------------------------------------------
    # Stage 1: within-Z projection → σ̂_ε
    # ------------------------------------------------------------------
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)                # (B, m, q, q)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)  # (B, m, q, q)
    mom_mask, z_rank, _, active_components, active_count = _groupZDiagnostics(ZtZ, mask_m, ns, q)
    G_mom_raw = mom_mask.sum(dim=1)                               # (B,)
    G_mom = G_mom_raw.clamp(min=1.0)                              # (B,)
    mom4 = mom_mask[:, :, None, None]                              # (B, m, 1, 1)
    enough_full_mom = G_mom_raw >= torch.maximum(
        active_count + 1.0, G_mom_raw.new_full((B,), float(d + 1))
    )
    enough_diag_mom = (G_mom_raw >= 2.0) & (active_count > 0)
    active_q = active_components.to(Zm.dtype)
    active_qq = active_q[:, :, None] * active_q[:, None, :]

    ZtZ_inv = _safeSolve(
        ZtZ_safe + _adaptiveRidgeBm(ZtZ_safe), eye_q_bm
    )                                                              # (B, m, q, q)

    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)                  # (B, m, q)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)                # (B, m, q, d)

    Zhat_y = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Zty)         # (B, m, q)
    Zhat_X = torch.einsum('bmqr,bmrd->bmqd', ZtZ_inv, ZtX)       # (B, m, q, d)
    My = (ym - torch.einsum('bmnq,bmq->bmn', Zm, Zhat_y)) * mask_n           # (B, m, n)
    MX = (Xm - torch.einsum('bmnq,bmqd->bmnd', Zm, Zhat_X)) * mask_n[..., None]  # (B, m, n, d)

    MXtMX = torch.einsum('bmnd,bmnk->bdk', MX, MX)               # (B, d, d)
    MXtMy = torch.einsum('bmnd,bmn->bd', MX, My)                  # (B, d)
    beta_wg = _safeSolve(MXtMX + _adaptiveRidge(MXtMX), MXtMy)  # (B, d)

    resid_M = My - torch.einsum('bmnd,bd->bmn', MX, beta_wg)     # (B, m, n)
    ss_w = resid_M.square().sum(dim=(1, 2))                       # (B,)
    mx_rank = _gramRank(MXtMX)
    df_w = (N - z_rank.sum(dim=1) - mx_rank).clamp(min=1.0)       # (B,)
    sigma_eps_sq = (ss_w / df_w).clamp(min=0.0)                   # (B,)

    # ------------------------------------------------------------------
    # Stage 2: group-level OLS b̂_g and MoM for Ψ
    # ------------------------------------------------------------------
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)                 # (B, d, d)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)                   # (B, d)
    beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)  # (B, d)
    resid_full = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_ols)) * mask_n  # (B, m, n)
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid_full)          # (B, m, q)
    bhat = torch.einsum('bmqr,bmr->bmq', ZtZ_inv, Ztr) * mask_m[:, :, None]  # (B, m, q) b̂_g

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat, bhat)        # (B, m, q, q)
    ZtZ_bhat = torch.einsum('bmqp,bmpk->bmqk', ZtZ_safe, bhat_outer)  # (B, m, q, q)

    mask4 = mask_m[:, :, None, None]                              # (B, m, 1, 1)
    sum_ZtZ = (ZtZ_safe * mom4).sum(dim=1)                        # (B, q, q) safe groups only
    sum_ZtZ_bhat = (ZtZ_bhat * mom4).sum(dim=1)                   # (B, q, q) safe groups only

    rhs_mom = sum_ZtZ_bhat - sigma_eps_sq[:, None, None] * G_mom[:, None, None] * eye_q
    Psi_raw = _safeSolve(sum_ZtZ + _adaptiveRidge(sum_ZtZ), rhs_mom)  # (B, q, q)

    # Per-component noise-corrected Psi diagonal floor to prevent EM getting stuck near Psi=0.
    # E[bhat_i²] ≈ Psi_ii + σ_ε² mean_g(ZtZ_inv_ii). Removing the noise term gives an
    # estimate of Psi_ii ≈ σ_rfx_i², which is near 0 when σ_rfx_i is small (so the floor
    # only activates for high-SNR components where MoM tends to under-estimate).
    # Applied per-component so inactive rfx dimensions (second Z column = 0 for q=1 datasets)
    # are not inflated — their signal_var ≈ 0 and the floor stays at 0.
    # Uses mom_mask to exclude near-singular groups (same as MoM sums above).
    mean_ZtZ_inv_diag = (ZtZ_inv.diagonal(dim1=-2, dim2=-1) * mom_mask[:, :, None]).sum(
        dim=1
    ) / G_mom[
        :, None
    ]                                       # (B, q)
    # Center bhat over informative groups before squaring so that a shared beta_ols offset
    # (bhat_g ≈ b_g + (beta_true − beta_ols)) cancels in the variance rather than inflating
    # signal_mean and psi_eig_cap. Without centering, the squared offset can be O(1) while
    # sigma_rfx² ≈ 0.2, loosening the cap by 10–20× and producing catastrophic EM spikes.
    bhat_mean = _maskedMean(bhat, mom_mask[:, :, None], dim=1)  # (B, q)
    bhat_centered = bhat - bhat_mean[:, None, :]                 # (B, m, q)
    bhat_signal = (
        bhat_centered.square() - sigma_eps_sq[:, None, None] * ZtZ_inv.diagonal(dim1=-2, dim2=-1)
    ).clamp(min=0.0)
    signal_mean = _maskedMean(bhat_signal, mom_mask[:, :, None], dim=1)
    signal_cap = (6.0 * signal_mean).clamp(min=sigma_eps_sq[:, None] * 1e-6)
    signal_winsor = torch.minimum(bhat_signal, signal_cap[:, None, :])
    signal_mean = _maskedMean(signal_winsor, mom_mask[:, :, None], dim=1)
    signal_median = _maskedMedian(signal_winsor, mom_mask[:, :, None], dim=1)
    psi_diag_signal = torch.minimum(signal_mean, 2.0 * signal_median)

    active_ns_mean = _maskedMean(ns, mask_m, dim=1).clamp(min=1.0)
    fallback_diag = (sigma_eps_sq / active_ns_mean).clamp(min=1e-10)[:, None].expand(B, q)
    fallback_diag = fallback_diag * active_q
    psi_diag_floor = torch.where(
        enough_diag_mom[:, None],
        torch.maximum(0.5 * psi_diag_signal * active_q, fallback_diag),
        fallback_diag,
    )
    psi_eig_cap = torch.maximum(
        (6.0 * psi_diag_signal * active_q).amax(dim=1),
        fallback_diag.amax(dim=1),
    )
    Psi_raw = Psi_raw + torch.diag_embed(
        (psi_diag_floor - Psi_raw.diagonal(dim1=-2, dim2=-1)).clamp(min=0.0)
    )                                                                   # bump diag to floor

    Psi_raw = 0.5 * (Psi_raw + Psi_raw.mT)
    Psi_raw = Psi_raw * active_qq
    Psi_raw = torch.where(enough_full_mom[:, None, None], Psi_raw, torch.diag_embed(psi_diag_floor))
    vals, vecs = _eighWithJitter(Psi_raw)                         # (B, q), (B, q, q)
    vals = vals.clamp(min=0.0)
    Psi = vecs @ torch.diag_embed(vals) @ vecs.mT                 # (B, q, q)
    Psi = _psdClampEigenvalues(Psi, psi_eig_cap)

    if uncorr is not None:
        Psi = torch.where(
            uncorr[:, None, None], torch.diag_embed(Psi.diagonal(dim1=-2, dim2=-1)), Psi
        )
        vals, vecs = _eighWithJitter(Psi)

    # ------------------------------------------------------------------
    # Stage 3: GLS β̂ via Woodbury
    # ------------------------------------------------------------------
    se2 = sigma_eps_sq.clamp(min=1e-12)

    # Ridge-regularized Psi_inv: (Psi + σ_ε²×1e-4×I)^{-1} reusing Stage-2 eigenvectors.
    # Pseudoinverse zeros eigenvalues ≈ 0 → Psi_inv≈0 → inner≈ZtZ → W_g=ZtZ_inv (OLS, no shrinkage).
    # Ridge ensures inv is large for small eigenvalues → strong shrinkage → BLUPs→0 for small Psi.
    # Effect is negligible when Psi eigenvalues >> σ_ε²×1e-4.
    inv_vals = 1.0 / (vals + se2[:, None] * 1e-4).clamp(min=1e-30)
    Psi_inv = vecs @ torch.diag_embed(inv_vals) @ vecs.mT        # (B, q, q)

    inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe  # (B, m, q, q)
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm)  # (B, m, q, q)
    W_g = W_g * mask4                                             # (B, m, q, q)

    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)                  # (B, m, d, q)

    W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)              # (B, m, q, d)
    correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)     # (B, d, d)
    W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)                # (B, m, q)
    correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)       # (B, d)

    inv_se2 = 1.0 / se2
    A_gls = inv_se2[:, None, None] * (XtX - correction_XX)         # (B, d, d)
    b_gls = inv_se2[:, None] * (Xty - correction_Xy)               # (B, d)

    # Detect GLS collapse: when correction_XX ≈ XtX (occurs when q ≥ d at high SNR,
    # because the random-effects space absorbs all between-group X variation), the
    # between-group information for β vanishes → A_gls → 0 → β_gls explodes and gets
    # clamped to ±50, corrupting residuals and BLUPs throughout the EM loop.
    # Fall back to the within-Z OLS estimate (β_wg) for those datasets: for q=d, β_wg=0
    # (no within-group information either), which at least keeps BLUPs well-behaved.
    xtx_max_diag = XtX.diagonal(dim1=-1, dim2=-2).amax(dim=-1).clamp(min=1.0)  # (B,)
    beta_identified = (XtX - correction_XX).diagonal(dim1=-1, dim2=-2).abs().amax(
        dim=-1
    ) > 1e-3 * xtx_max_diag  # (B,) True = between-group information is non-negligible

    A_gls_reg = A_gls + _adaptiveRidge(A_gls)
    beta_gls = _safeSolve(A_gls_reg, b_gls)                        # (B, d)

    # clamp before BLUP computation: near-cancellation in A_gls=(XtX−correction_XX) when
    # Ψ≈0 produces finite-but-huge values that nan_to_num cannot catch.
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
    beta_gls = torch.where(beta_identified[:, None], beta_gls, beta_wg)
    Psi = Psi.nan_to_num(nan=0.0, posinf=0.0)

    # BLUPs
    resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
    Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)           # (B, m, q)
    blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)              # (B, m, q)
    # clamp: W_g can be huge when ZtZ is near-singular (small n_g, q=2) even with beta clamped
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    # Posterior variance: Cov(b_g | data) = σ_ε² · W_g  (diagonal = marginal var)
    # cap at 25 (std ≤ 5) — W_g can be large when ZtZ near-singular (small n_g, q=2)
    blup_var = (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
    blup_var = blup_var.nan_to_num(nan=0.0, posinf=0.0)              # (B, m, q)

    # EM iterations: jointly update Ψ, σ_ε², and β̂_GLS.
    # M-step: Ψ = mean_g(b̂_g b̂_g' + σ²_ε W_g) — exact E[b_g b_g'|data] for Gaussian b_g,
    #         σ_ε² = RSS/(N−d−T) (REML-like, T = Σ_g tr(ZtZ_g W_g) effective df).
    # E-step: W_g via ridge-regularized Ψ⁻¹, then β̂_GLS and BLUPs under updated parameters.
    beta_rank = mx_rank.clamp(min=1.0, max=float(d))
    for _ in range(n_em):
        # M-step: Ψ using full posterior covariance (exact for Gaussian b_g)
        blup_outer = torch.einsum('bmq,bmr->bmqr', blups, blups)     # (B, m, q, q)
        post_cov = se2[:, None, None, None] * W_g                    # (B, m, q, q)  σ²_ε W_g
        Psi_em = _psdProject(
            ((blup_outer + post_cov) * mom4).sum(dim=1) / G_mom[:, None, None]
        )  # (B, q, q)
        psi_diag_em = Psi_em.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)
        Psi_em = torch.where(enough_full_mom[:, None, None], Psi_em, torch.diag_embed(psi_diag_em))
        Psi = _psdClampEigenvalues(_psdProject((0.5 * Psi + 0.5 * Psi_em) * active_qq), psi_eig_cap)
        if uncorr is not None:
            Psi = torch.where(
                uncorr[:, None, None], torch.diag_embed(Psi.diagonal(dim1=-2, dim2=-1)), Psi
            )

        # M-step: σ_ε² (REML-like df correction using current blups and beta_gls)
        resid_em = (
            ym
            - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)
            - torch.einsum('bmnq,bmq->bmn', Zm, blups)
        ) * mask_n
        ss_em = resid_em.square().sum(dim=(1, 2))                    # (B,)
        ZtZ_W = torch.einsum('bmqr,bmrs->bmqs', ZtZ_safe, W_g)      # (B, m, q, q)
        T = (ZtZ_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1) * mask_m).sum(dim=1)  # (B,)
        # Cap T so the REML denominator N−d−T stays ≥ 10% of N−d, preventing blow-up
        # when T ≈ N−d (λ_g → 1 for all groups at high SNR or near-singular ZtZ_g).
        T_safe = T.clamp(max=0.9 * (N - beta_rank).clamp(min=1.0))
        se2 = (ss_em / (N - beta_rank - T_safe).clamp(min=1.0)).clamp(min=1e-12)

        # E-step: W_g via ridge-regularized Ψ⁻¹ using updated Ψ and se2
        psi_ridge = se2[:, None, None] * 1e-4 * eye_q
        psi_reg = Psi + psi_ridge
        vals_r, vecs_r = _eighWithJitter(psi_reg)
        Psi_inv = vecs_r @ torch.diag_embed(1.0 / vals_r.clamp(min=1e-30)) @ vecs_r.mT
        inner = se2[:, None, None, None] * Psi_inv[:, None] + ZtZ_safe  # (B, m, q, q)
        W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm) * mask4

        # E-step: β̂_GLS and Ztr_gls under updated W_g and se2
        inv_se2 = 1.0 / se2
        W_ZtX = torch.einsum('bmqp,bmpd->bmqd', W_g, ZtX)
        correction_XX = torch.einsum('bmdq,bmqk->bdk', XtZ, W_ZtX)
        W_Zty = torch.einsum('bmqp,bmp->bmq', W_g, Zty)
        correction_Xy = torch.einsum('bmdq,bmq->bd', XtZ, W_Zty)
        A_gls = inv_se2[:, None, None] * (XtX - correction_XX)
        b_gls = inv_se2[:, None] * (Xty - correction_Xy)
        A_gls_reg = A_gls + _adaptiveRidge(A_gls)
        beta_gls = _safeSolve(A_gls_reg, b_gls)
        beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        beta_gls = torch.where(beta_identified[:, None], beta_gls, beta_wg)
        resid_gls = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_gls)) * mask_n
        Ztr_gls = torch.einsum('bmnq,bmn->bmq', Zm, resid_gls)

        # E-step: BLUPs and posterior variance
        blups = torch.einsum('bmqp,bmp->bmq', W_g, Ztr_gls)
        blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        blup_var = (
            (se2[:, None, None, None] * W_g).diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)
        )
        blup_var = blup_var.nan_to_num(nan=0.0, posinf=0.0)

    # Inflate blup_var to account for uncertainty in the Psi estimate.
    # Var[Psi] ∝ Psi²/(G-d), delta-method gives 1 + 2/(G-d).
    # Clamp denominator at 4 to cap inflation at 50% for large G; uncapped it over-inflates
    # blup_var in real-data regimes where Psi is well-estimated (observed ratio < 0.5).
    df_sigma = (G - d).clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma.clamp(min=4.0))[:, None, None]

    # Kackar-Harville correction: blup_var = diag(σ²W_g) conditions on β as known, but actual
    # BLUP error also includes W_g Z^T X (β_hat - β). Dominant for large groups where W_g→Ψ⁻¹,
    # (1-λ)Ψ→0.
    # beta_var uses a truncated eigendecomposition of A_gls_reg: near-zero eigenvalues (rank
    # deficiency from m<d, collinear group means, or q≥d collapse) inflate beta_var_kh via
    # the adaptive-ridge reciprocal ≈1e6. Truncating below 1e-3 × max_eig caps beta_var at
    # the identified directions only, zeroing contributions from the null space.
    vals_kh, vecs_kh = torch.linalg.eigh(A_gls_reg)                # (B, d), (B, d, d)
    max_kh = vals_kh.amax(dim=-1, keepdim=True).clamp(min=1.0)     # (B, 1)
    inv_vals_kh = torch.where(
        vals_kh > 1e-3 * max_kh,
        1.0 / vals_kh.clamp(min=1e-30),
        torch.zeros_like(vals_kh),
    )
    beta_var_kh = (vecs_kh**2 * inv_vals_kh[:, None, :]).sum(dim=-1).clamp(min=1e-8)  # (B, d)
    kh_corr = (W_ZtX**2 * beta_var_kh[:, None, None, :]).sum(dim=-1)  # (B, m, q)
    # Zero KH for collapsed GLS (q≥d or G<d): those datasets use beta_wg so β-uncertainty
    # is not meaningful; truncation above handles the collinear-group-means case.
    gls_determined = G >= float(d)  # (B,)
    kh_corr = kh_corr * (beta_identified & gls_determined)[:, None, None]
    blup_var = blup_var + kh_corr

    # Floor blup_var at Psi_diag / (2 * n_g): prevents near-zero declared variance for
    # small groups on real (sampled) data where the Gaussian model may be misspecified.
    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)  # (B, q)
    blup_var_floor = psi_diag[:, None, :] / (2.0 * ns.clamp(min=1.0)[:, :, None])
    blup_var = blup_var.clamp(min=blup_var_floor)

    sigma_rfx = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()          # (B, q)
    sigma_eps_1d = se2.clamp(min=0.0).sqrt().nan_to_num(nan=1.0, posinf=1.0)  # (B,)

    ns_f_loc = ns.clamp(min=1.0)                                              # (B, m)
    resid_g = (resid_gls.sum(dim=2) / ns_f_loc * mask_m).unsqueeze(-1)       # (B, m, 1)
    resid_g = resid_g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    beta_wg_out = beta_wg.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    # Clamp bhat output: pathological groups (n_g ≈ q) still have bhat blown up via ZtZ_inv,
    # but they were excluded from MoM so Psi is clean. Cap output to ±10 for NN input safety.
    bhat_out = bhat.clamp(-10.0, 10.0).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # (B, m, q)

    return {
        'beta_est': beta_gls,  # (B, d)
        'beta_var': beta_var_kh,  # (B, d)
        'beta_wg': beta_wg_out,  # (B, d)
        'sigma_eps_est': sigma_eps_1d.unsqueeze(-1),  # (B, 1)
        'sigma_rfx_est': sigma_rfx,  # (B, q)
        'blup_est': blups,  # (B, m, q)
        'blup_var': blup_var,  # (B, m, q)
        'bhat': bhat_out,  # (B, m, q)
        'resid_g': resid_g,  # (B, m, 1)
        'Psi': Psi,  # (B, q, q)
    }


# ---------------------------------------------------------------------------
# Public normal-LMM dispatcher
# ---------------------------------------------------------------------------


def lmmNormal(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)
    mask_m: torch.Tensor,  # (B, m)
    ns: torch.Tensor,  # (B, m)
    n_total: torch.Tensor,  # (B,)
    n_em: int = 3,
    uncorr: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Closed-form GLS for the Gaussian LMM."""
    return _lmmNormalFull(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        n_em=max(n_em, _NORMAL_FULL_MIN_EM),
        uncorr=uncorr,
        mask_q=mask_q,
    )


# ---------------------------------------------------------------------------
# Private GLMM implementation
# ---------------------------------------------------------------------------


def _lmmGlmm(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
    mask_n: torch.Tensor,  # (B, m, n)  1 for active observations
    mask_m: torch.Tensor,  # (B, m)     1 for active groups
    ns: torch.Tensor,  # (B, m)     group sizes (float)
    n_total: torch.Tensor,  # (B,)       total active observations
    likelihood_family: int,
    n_newton: int = 3,
    uncorr: torch.Tensor | None = None,  # (B,) bool — force Ψ diagonal for these datasets
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM variance-component estimator (private).

    Parameters
    ----------
    Xm, ym, Zm  : batched grouped data tensors
    mask_n      : observation-level binary mask (B, m, n)
    mask_m      : group-level binary mask (B, m)
    ns          : per-group observation counts (B, m)
    n_total     : per-dataset total observation count (B,)
    likelihood_family : 1=Bernoulli/logit, 2=Poisson/log
    n_newton    : damped Newton steps for the per-group Laplace mode

    Returns
    -------
    dict with keys: beta_est, beta_wg, sigma_rfx_est, blup_est, blup_var,
                    bhat, resid_g, phi_pearson, psi_0, Psi_pql, Psi_lap, mean_Hg_inv
    """
    B, m, n, d = Xm.shape
    q = Zm.shape[-1]
    N = n_total.float()                                # (B,)
    G = mask_m.sum(dim=1).clamp(min=1.0)              # (B,)
    active = mask_m.bool()                             # (B, m)
    mask4 = mask_m[:, :, None, None]                  # (B, m, 1, 1)

    eye_q = torch.eye(q, device=Xm.device, dtype=Xm.dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)

    # ------------------------------------------------------------------
    # Stage 0: pooled IRLS → β₀, overdispersion φ, scale-0 estimate ψ₀
    # ------------------------------------------------------------------
    if likelihood_family == 1:
        beta_0 = irlsBernoulliCompacted(Xm, ym, mask_n)
    else:
        beta_0 = irlsPoissonCompacted(Xm, ym, mask_n)

    eta_0 = torch.einsum('bmnd,bd->bmn', Xm, beta_0)  # (B, m, n)

    if likelihood_family == 1:
        mu_0 = torch.sigmoid(eta_0)
        pearson = (ym - mu_0).square() / (mu_0 * (1.0 - mu_0)).clamp(min=1e-6)
    else:
        mu_0, deriv_0 = _poissonMeanDerivative(eta_0)
        pearson = (ym - mu_0).square() / mu_0.clamp(min=1e-6)

    phi_pearson = (pearson * mask_n).sum(dim=(1, 2)) / (N - d).clamp(min=1.0)  # (B,)
    mu_bar = (mu_0 * mask_n).sum(dim=(1, 2)) / N.clamp(min=1.0)                # (B,)

    if likelihood_family == 2:
        psi_0 = ((phi_pearson - 1.0) / mu_bar.clamp(min=1e-6)).clamp(min=0.0)
    else:
        n_bar = N / G
        rho_hat = ((phi_pearson - 1.0) / (n_bar - 1.0).clamp(min=1.0)).clamp(min=0.0)
        psi_0 = (rho_hat / (mu_bar * (1.0 - mu_bar)).clamp(min=1e-6)).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Stage 1: one PQL pass at (β₀, b=0) → ZWZ⁻¹, b̂_OLS, Ψ̂_PQL
    # ------------------------------------------------------------------
    w1, _ = _pqlWorking(eta_0, ym, mask_n, likelihood_family)  # bg=0, so eta = eta_0

    ZWZ = torch.einsum('bmnq,bmn,bmnr->bmqr', Zm, w1, Zm)          # (B, m, q, q)

    ZWZ_safe = torch.where(active[:, :, None, None], ZWZ, eye_q)
    ZWZ_inv = _safeSolve(ZWZ_safe + _adaptiveRidgeBm(ZWZ_safe), eye_q_bm)  # (B, m, q, q)

    if likelihood_family == 2:
        score_0 = (ym - mu_0) * deriv_0
    else:
        score_0 = ym - mu_0
    ZtYmMu = torch.einsum('bmnq,bmn->bmq', Zm, score_0 * mask_n)  # (B, m, q)
    bhat_ols = torch.einsum('bmqr,bmr->bmq', ZWZ_inv, ZtYmMu) * mask_m[:, :, None]

    bhat_outer = torch.einsum('bmq,bmr->bmqr', bhat_ols, bhat_ols)       # (B, m, q, q)
    mean_bhat_outer = (bhat_outer * mask4).sum(dim=1) / G[:, None, None]
    mean_ZWZ_inv = (ZWZ_inv * mask4).sum(dim=1) / G[:, None, None]
    Psi_pql = _psdProject(mean_bhat_outer - mean_ZWZ_inv)                 # (B, q, q)

    # Floor eigenvalues at psi_0 (overdispersion-based RE scale estimate).
    # Prevents degenerate near-zero Ψ̂_PQL after PSD projection, which would
    # make Ψ̂_PQL⁻¹ → ∞ and kill Newton updates.
    vals_pql, vecs_pql = _eighWithJitter(Psi_pql)
    vals_pql = vals_pql.clamp(min=psi_0[:, None])
    Psi_pql = vecs_pql @ torch.diag_embed(vals_pql) @ vecs_pql.mT        # (B, q, q)
    if uncorr is not None:
        Psi_pql = torch.where(
            uncorr[:, None, None], torch.diag_embed(Psi_pql.diagonal(dim1=-2, dim2=-1)), Psi_pql
        )

    # ------------------------------------------------------------------
    # Stage 2: alternating Newton–GLS loop, up to max_passes=6.
    # Pass 1 cold-starts Newton under pooled β₀ and a stabilized Ψ̂_PQL inverse.
    # Passes 2–max_passes warm-start from previous BLUPs under β̂_GLS and
    # ridge-inv(Ψ̂_Lap); early exit when the 95th-percentile change in both
    # β and diag(Ψ̂) falls below 1e-3.
    # ------------------------------------------------------------------
    pass_args = (
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        mask4,
        active,
        eye_q,
        eye_q_bm,
        G,
        likelihood_family,
        n_newton,
    )
    if likelihood_family == 1:
        psi_0_floor = psi_0.clamp(min=_BERNOULLI_INITIAL_PSI_FLOOR)
    elif likelihood_family == 2:
        psi_0_floor = psi_0.clamp(min=_POISSON_INITIAL_PSI_FLOOR)
    else:
        psi_0_floor = psi_0.clamp(min=1e-6)
    max_passes = 6

    def _sanitize(b_gls: torch.Tensor, psi: torch.Tensor):
        """Clamp beta_gls and clean Psi_lap before using as next-pass inputs."""
        beta_clamp = _POISSON_BETA_CLAMP if likelihood_family == 2 else 50.0
        return (
            b_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-beta_clamp, beta_clamp),
            psi.nan_to_num(nan=0.0, posinf=0.0),
        )

    # Pass 1: pooled β₀, cold start.  Discrete outcomes can make Psi_pql exactly
    # zero in weakly identified directions; use a weak variance floor for shrinkage.
    Psi_inv = (
        _ridgeInv(Psi_pql, psi_0_floor) if likelihood_family in (1, 2) else _pseudoInverse(Psi_pql)
    )
    beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls, blup_kh_var = _pqlPass(
        beta_0, Psi_inv, *pass_args
    )
    beta_gls, Psi_lap = _sanitize(beta_gls, Psi_lap)
    if uncorr is not None:
        Psi_lap = torch.where(
            uncorr[:, None, None], torch.diag_embed(Psi_lap.diagonal(dim1=-2, dim2=-1)), Psi_lap
        )

    # Passes 2–max_passes: warm start, ridge-regularized Ψ̂_Lap, until convergence.
    # Each pass refines (β, b̂_g, Ψ̂_Lap) jointly. Early exit when the 95th-percentile
    # change across the batch is below tolerance — avoids running all max_passes when a
    # few pathological datasets (small m) never satisfy the strict amax criterion.
    for _ in range(max_passes - 1):
        beta_prev = beta_gls
        psi_diag_prev = Psi_lap.diagonal(dim1=-2, dim2=-1)

        Psi_inv = _ridgeInv(Psi_lap, psi_0_floor)
        beta_gls, Psi_lap, blups, Kg_inv, mean_Hg_inv, resid_gls, blup_kh_var = _pqlPass(
            beta_gls, Psi_inv, *pass_args, bg_init=blups
        )
        beta_gls, Psi_lap = _sanitize(beta_gls, Psi_lap)
        if uncorr is not None:
            Psi_lap = torch.where(
                uncorr[:, None, None], torch.diag_embed(Psi_lap.diagonal(dim1=-2, dim2=-1)), Psi_lap
            )

        d_beta = (beta_gls - beta_prev).abs().amax(dim=-1)                               # (B,)
        d_psi = (Psi_lap.diagonal(dim1=-2, dim2=-1) - psi_diag_prev).abs().amax(dim=-1)  # (B,)
        if torch.quantile(d_beta, 0.95) < 1e-3 and torch.quantile(d_psi, 0.95) < 1e-3:
            break

    # ------------------------------------------------------------------
    # Pack outputs
    # ------------------------------------------------------------------
    beta_gls = beta_gls.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    blups = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    Psi_lap = Psi_lap.nan_to_num(nan=0.0, posinf=0.0)
    if likelihood_family == 1:
        corr_alpha = G / (G + _BERNOULLI_CORR_SHRINKAGE_C)
        Psi_lap = _shrinkOffDiagonal(Psi_lap, corr_alpha)
    elif likelihood_family == 2:
        corr_alpha = G / (G + _POISSON_CORR_SHRINKAGE_C)
        Psi_lap = _shrinkOffDiagonal(Psi_lap, corr_alpha)
    Psi_pql = Psi_pql.nan_to_num(nan=0.0, posinf=0.0)
    mean_Hg_inv = mean_Hg_inv.nan_to_num(nan=0.0, posinf=0.0)
    sigma_rfx_est = Psi_lap.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt().nan_to_num()

    # Per-group posterior variance: diagonal of K_g^{-1} = (ZWZ + Ψ̂_Lap^{-1})^{-1}.
    # Uses the GLS-consistent Ψ̂_Lap precision (not the ridge-inflated Newton Ψ⁻¹),
    # mirroring the Normal path's σ²_ε · W_g.  Cap at 25 (std ≤ 5).
    blup_var = Kg_inv.diagonal(dim1=-2, dim2=-1).clamp(min=0.0, max=25.0)  # (B, m, q)
    blup_var = (blup_var * mask_m[:, :, None]).nan_to_num(nan=0.0, posinf=0.0)

    # Inflate blup_var to account for uncertainty in Ψ̂_Lap (same rationale as Normal path).
    # Use G as denominator (no d subtraction) since PQL doesn't have an explicit df formula.
    df_sigma = G.clamp(min=1.0)
    blup_var = blup_var * (1.0 + 2.0 / df_sigma)[:, None, None]
    blup_var = blup_var + blup_kh_var
    if likelihood_family == 1:
        blup_var = blup_var * _BERNOULLI_BLUP_VAR_INFLATION
        if d > 8:
            blup_var = blup_var + _BERNOULLI_HIGH_D_BLUP_VAR_FLOOR * mask_m[:, :, None]

    # Per-group mean working residual (after removing fixed effects)
    resid_g = (resid_gls.sum(dim=2) / ns.clamp(min=1.0) * mask_m).unsqueeze(-1)  # (B, m, 1)
    resid_g = resid_g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    beta_wg_out = beta_0.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # pooled IRLS (no rfx)
    bhat_out = bhat_ols.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)   # (B, m, q)

    return {
        'beta_est': beta_gls,  # (B, d)
        'beta_wg': beta_wg_out,  # (B, d)  pooled IRLS (no-rfx analog of beta_wg)
        'sigma_rfx_est': sigma_rfx_est,  # (B, q)
        'blup_est': blups,  # (B, m, q)
        'blup_var': blup_var,  # (B, m, q)
        'bhat': bhat_out,  # (B, m, q)
        'resid_g': resid_g,  # (B, m, 1)
        'phi_pearson': phi_pearson,  # (B,)
        'psi_0': psi_0,  # (B,)
        'Psi_pql': Psi_pql,  # (B, q, q)
        'Psi_lap': Psi_lap,  # (B, q, q)
        'mean_Hg_inv': mean_Hg_inv,  # (B, q, q)
    }


# ---------------------------------------------------------------------------
# Public GLMM dispatchers
# ---------------------------------------------------------------------------


def lmmBernoulli(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    n_newton: int = 3,
    uncorr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM for Bernoulli/logit outcomes."""
    return _lmmGlmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=1,
        n_newton=n_newton,
        uncorr=uncorr,
    )


def lmmPoisson(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    ns: torch.Tensor,
    n_total: torch.Tensor,
    n_newton: int = 3,
    uncorr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """PQL-based GLMM for Poisson/log outcomes."""
    return _lmmGlmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=2,
        n_newton=n_newton,
        uncorr=uncorr,
    )


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


# ---------------------------------------------------------------------------
# Analytical BLUP context for the local-flow conditioning signal
# ---------------------------------------------------------------------------


def analyticalBLUPContext(
    data: dict[str, torch.Tensor],
    beta: torch.Tensor,  # (B, S, d) fixed effects — constrained scale
    sigma_rfx: torch.Tensor,  # (B, S, q) — constrained scale
    sigma_eps: torch.Tensor,  # (B, S)   — constrained scale
    z_corr: torch.Tensor | None,  # (B, S, d_corr) unconstrained atanh or None
    clamp: float = 20.0,
) -> torch.Tensor:
    """Analytical BLUP mean, marginal std, and shrinkage given global parameter samples.

    Uses the closed-form Gaussian posterior (family == 0 only).
    Non-normal GLMMs would need per-sample Newton/Laplace solves for every group
    rather than a closed form, so they currently use the fixed PQL BLUP stats.
    Returns (B, m, S, 3*q).
    """
    from metabeta.posthoc.gaussian_local import analyticalBLUPStats

    q = sigma_rfx.shape[-1]
    d = beta.shape[-1]

    Sigma_rfx_inv: torch.Tensor | None = None
    if z_corr is not None:
        L_corr = unconstrainedToCholesky(z_corr, q)                    # (B, S, q, q)
        sr_inv_diag = torch.diag_embed(1.0 / sigma_rfx.clamp(min=1e-6))  # (B, S, q, q)
        A = torch.linalg.solve_triangular(L_corr, sr_inv_diag, upper=False)
        Sigma_rfx_inv = A.mT @ A                                        # (B, S, q, q)

    mu, blup_std, lambda_g = analyticalBLUPStats(
        data['y'],
        data['X'][..., :d],
        data['Z'][..., :q],
        beta,
        sigma_rfx,
        sigma_eps,
        data['mask_n'],
        Sigma_rfx_inv=Sigma_rfx_inv,
    )
    return torch.cat([mu.clamp(-clamp, clamp), blup_std.clamp(max=clamp), lambda_g], dim=-1)
