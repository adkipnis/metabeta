import math
import logging
import numpy as np
import torch
from scipy.stats import expon, norm, t
from torch import distributions as D

from metabeta.utils.constants import (
    FFX_FAMILIES,
    SIGMA_FAMILIES,
    FFX_FAMILY_PROBS,
    SIGMA_RFX_FAMILY_PROBS,
    SIGMA_EPS_FAMILY_PROBS,
    STUDENT_DF,
    LIKELIHOOD_FAMILIES,
    LIKELIHOOD_HAS_SIGMA_EPS,
    LIKELIHOOD_BAMBI_FAMILY,
    hasSigmaEps,
    bambiFamilyName,
)
from metabeta.utils.linalg import choleskySolve
from metabeta.utils.preprocessing import standardize
from metabeta.utils.regularization import unconstrainedToCholesky

logger = logging.getLogger(__name__)

# numerical stabilization constants
BERNOULLI_LOGIT_CLIP_ABS = 500.0
BERNOULLI_ETA_ABS_MAX = 10.0
BERNOULLI_REROLL_EXTREME_FRACTION_MAX = 0.05
BERNOULLI_REROLL_MAX_ATTEMPTS = 20
BERNOULLI_LP_SD_CAP_LOW = 2.0
BERNOULLI_LP_SD_CAP_HIGH = 4.0
POISSON_ETA_CLIP_MAX = 10.0
POISSON_X_CLIP_ABS = 5.0
POISSON_REROLL_CLIP_FRACTION_MAX = 0.01
POISSON_REROLL_MAX_ATTEMPTS = 20
POISSON_LP_SD_CAP_LOW = 1.0
POISSON_LP_SD_CAP_HIGH = 2.0
CLIP_WARN_FRACTION = 0.05
CLIP_WARN_MAX_MESSAGES = 10

_CLIP_WARN_COUNTS: dict[str, int] = {}


def _warnIfFrequentClipping(name: str, frac_clipped: float) -> None:
    if frac_clipped < CLIP_WARN_FRACTION:
        return
    count = _CLIP_WARN_COUNTS.get(name, 0)
    if count >= CLIP_WARN_MAX_MESSAGES:
        return
    _CLIP_WARN_COUNTS[name] = count + 1
    logger.warning(
        'Frequent clipping in %s: %.2f%% values clipped (warning %d/%d).',
        name,
        100.0 * frac_clipped,
        count + 1,
        CLIP_WARN_MAX_MESSAGES,
    )


# ---------------------------------------------------------------------------
# NumPy sampling
# ---------------------------------------------------------------------------


def _sampleNormalNp(
    rng: np.random.Generator, loc: np.ndarray, scale: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return norm(loc, scale).rvs(size=size, random_state=rng)


def _sampleStudentNp(
    rng: np.random.Generator, loc: np.ndarray, scale: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return t(df=STUDENT_DF, loc=loc, scale=scale).rvs(size=size, random_state=rng)


def _sampleHalfNormalNp(
    rng: np.random.Generator, scale: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return np.abs(norm(0, scale).rvs(size=size, random_state=rng))


def _sampleHalfStudentNp(
    rng: np.random.Generator, scale: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return np.abs(t(df=STUDENT_DF, loc=0, scale=scale).rvs(size=size, random_state=rng))


def _sampleExponentialNp(
    rng: np.random.Generator, scale: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return expon(scale=scale).rvs(size=size, random_state=rng)


_FFX_SAMPLE_NP = (_sampleNormalNp, _sampleStudentNp)
_SIGMA_SAMPLE_NP = (_sampleHalfNormalNp, _sampleHalfStudentNp, _sampleExponentialNp)


def sampleFfxNp(
    family: int,
    loc: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
    size: tuple[int, ...],
) -> np.ndarray:
    sample_fn = _FFX_SAMPLE_NP[family]
    return sample_fn(rng, loc, scale, size)


def sampleSigmaNp(
    family: int,
    scale: np.ndarray,
    rng: np.random.Generator,
    size: tuple[int, ...],
) -> np.ndarray:
    sample_fn = _SIGMA_SAMPLE_NP[family]
    return sample_fn(rng, scale, size)


# ---------------------------------------------------------------------------
# Torch log-prob (importance sampling)
# ---------------------------------------------------------------------------


def _logProbNormal(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return D.Normal(loc, scale).log_prob(x)


def _logProbStudent(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return D.StudentT(df=STUDENT_DF, loc=loc, scale=scale).log_prob(x)


def _logProbHalfNormal(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return D.HalfNormal(scale=scale).log_prob(x)


def _logProbHalfStudent(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return D.StudentT(df=STUDENT_DF, loc=0, scale=scale).log_prob(x) + math.log(2.0)


def _logProbExponential(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rate = 1.0 / (scale + 1e-12)
    return D.Exponential(rate=rate).log_prob(x)


_FFX_LOG_PROB = (_logProbNormal, _logProbStudent)
_SIGMA_LOG_PROB = (_logProbHalfNormal, _logProbHalfStudent, _logProbExponential)


def logProbFfx(
    x: torch.Tensor,  # (b, s, d)
    loc: torch.Tensor,  # (b, 1, d)
    scale: torch.Tensor,  # (b, 1, d)
    family: torch.Tensor,  # (b,)
    mask: torch.Tensor | None = None,  # (b, 1, d)
) -> torch.Tensor:
    """Batched log-prob for ffx families. Returns (b, s)."""
    b, s = x.shape[:2]
    out = x.new_zeros(b, s)
    for i, fn in enumerate(_FFX_LOG_PROB):
        sel = family == i
        if not sel.any():
            continue
        lp = fn(x[sel], loc[sel], scale[sel])
        if mask is not None:
            lp = lp * mask[sel]
        out[sel] = lp.sum(-1)
    return out


def logProbSigma(
    x: torch.Tensor,  # (b, s, q) or (b, s)
    scale: torch.Tensor,  # (b, 1, q) or (b, 1)
    family: torch.Tensor,  # (b,)
    mask: torch.Tensor | None = None,  # (b, 1, q) or None
) -> torch.Tensor:
    """Batched log-prob for sigma families. Returns (b, s)."""
    b, s = x.shape[:2]
    out = x.new_zeros(b, s)
    for i, fn in enumerate(_SIGMA_LOG_PROB):
        sel = family == i
        if not sel.any():
            continue
        lp = fn(x[sel], scale[sel])
        if mask is not None:
            lp = lp * mask[sel]
        if lp.dim() > 2:
            lp = lp.sum(-1)
        out[sel] = lp
    return out


# ---------------------------------------------------------------------------
# Torch sampling (prior predictive) - select approach
# ---------------------------------------------------------------------------


def _sampleNormalTorch(
    loc: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    return D.Normal(loc, scale).sample(shape)


def _sampleStudentTorch(
    loc: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    return D.StudentT(df=STUDENT_DF, loc=loc, scale=scale).sample(shape)


def _sampleHalfNormalTorch(scale: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return D.HalfNormal(scale=scale).sample(shape)  # type: ignore


def _sampleHalfStudentTorch(scale: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return D.StudentT(df=STUDENT_DF, scale=scale).sample(shape).abs()


def _sampleExponentialTorch(scale: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    rate = 1.0 / (scale + 1e-12)
    return D.Exponential(rate=rate).sample(shape)


_FFX_SAMPLE_TORCH = (_sampleNormalTorch, _sampleStudentTorch)
_SIGMA_SAMPLE_TORCH = (
    _sampleHalfNormalTorch,
    _sampleHalfStudentTorch,
    _sampleExponentialTorch,
)


def sampleFfxTorch(
    loc: torch.Tensor,  # (b, d)
    scale: torch.Tensor,  # (b, d)
    family: torch.Tensor,  # (b,)
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Sample from ffx prior families. Returns (*shape, b, d)."""
    out = loc.new_zeros(*shape, *loc.shape)
    for i, fn in enumerate(_FFX_SAMPLE_TORCH):
        sel = family == i
        if not sel.any():
            continue
        out[..., sel, :] = fn(loc[sel], scale[sel], shape)
    return out


def sampleSigmaTorch(
    scale: torch.Tensor,  # (b, q) or (b,)
    family: torch.Tensor,  # (b,)
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Sample from sigma prior families. Returns (*shape, b, q) or (*shape, b)."""
    out = scale.new_zeros(*shape, *scale.shape)
    for i, fn in enumerate(_SIGMA_SAMPLE_TORCH):
        sel = family == i
        if not sel.any():
            continue
        if scale.dim() > 1:
            out[..., sel, :] = fn(scale[sel], shape)
        else:
            out[..., sel] = fn(scale[sel], shape)
    return out


# ---------------------------------------------------------------------------
# NumPy likelihood simulation
# ---------------------------------------------------------------------------


def _expit(x: np.ndarray) -> np.ndarray:
    lo, hi = -BERNOULLI_LOGIT_CLIP_ABS, BERNOULLI_LOGIT_CLIP_ABS
    clipped = np.clip(x, lo, hi)
    frac = float(np.mean((x < lo) | (x > hi)))
    _warnIfFrequentClipping('bernoulli_logits_np', frac)
    return 1.0 / (1.0 + np.exp(-clipped))


def simulateNormalNp(
    rng: np.random.Generator,
    eta: np.ndarray,
    sigma_eps: float,
) -> np.ndarray:
    return rng.normal(loc=eta, scale=sigma_eps).astype(eta.dtype)


def simulateBernoulliNp(
    rng: np.random.Generator,
    eta: np.ndarray,
    sigma_eps: float = 0.0,  # stand-in
) -> np.ndarray:
    p = _expit(eta)
    return rng.binomial(1, p).astype(eta.dtype)


def simulatePoissonNp(
    rng: np.random.Generator,
    eta: np.ndarray,
    sigma_eps: float = 0.0,
) -> np.ndarray:
    frac = float(np.mean(eta > POISSON_ETA_CLIP_MAX))
    _warnIfFrequentClipping('poisson_eta_np', frac)
    rate = np.exp(np.clip(eta, max=POISSON_ETA_CLIP_MAX))
    return rng.poisson(rate).astype(eta.dtype)


_LIKELIHOOD_SIMULATE_NP = (simulateNormalNp, simulateBernoulliNp, simulatePoissonNp)


def simulateYNp(
    rng: np.random.Generator,
    eta: np.ndarray,
    sigma_eps: float,
    likelihood_family: int = 0,
) -> np.ndarray:
    """Sample y given linear predictor eta and likelihood family."""
    return _LIKELIHOOD_SIMULATE_NP[likelihood_family](rng, eta, sigma_eps)


# ---------------------------------------------------------------------------
# Fixed-family log-probs (rfx prior and likelihood)
# ---------------------------------------------------------------------------


def logProbRfx(
    rfx: torch.Tensor,  # (b, m, s, q)
    sigma_rfx: torch.Tensor,  # (b, s, q)
    mask: torch.Tensor | None = None,  # (b, m, 1, q)
) -> torch.Tensor:
    """Log-prior for rfx (always Normal)."""
    scale = sigma_rfx.unsqueeze(1) + 1e-12
    lp = D.Normal(loc=0, scale=scale).log_prob(rfx)
    if mask is not None:
        lp = lp * mask
    return lp.sum(dim=(1, -1))  # (b, s)


def logProbRfxCorrelated(
    rfx: torch.Tensor,  # (b, m, s, q)
    sigma_rfx: torch.Tensor,  # (b, s, q)
    L_corr: torch.Tensor,  # (b, s, q, q) lower-triangular Cholesky of correlation matrix
    mask: torch.Tensor | None = None,  # (b, m, 1, q)
) -> torch.Tensor:
    """Log-prior for rfx under correlated Normal: rfx_i ~ MVN(0, diag(σ) @ C @ diag(σ)).

    Uses the Cholesky factor of C (L_corr) to build the covariance Cholesky efficiently.
    Returns (b, s).
    """
    q = L_corr.shape[-1]
    # unsqueeze m dim so the distribution broadcasts over groups: (b, 1, s, q, q)
    L_cov = sigma_rfx.unsqueeze(1).unsqueeze(-1) * L_corr.unsqueeze(1) + 1e-12 * torch.eye(
        q, dtype=L_corr.dtype, device=L_corr.device
    )
    loc = torch.zeros(*L_cov.shape[:-1], dtype=L_corr.dtype, device=L_corr.device)
    dist = D.MultivariateNormal(loc=loc, scale_tril=L_cov)  # batch (b, 1, s), event (q,)
    lp = dist.log_prob(rfx)  # rfx (b, m, s, q) -> lp (b, m, s)
    if mask is not None:
        # mask shape (b, m, 1, q) - reduce to (b, m, s) by checking any active q dim
        active = mask[..., 0].expand_as(lp)  # (b, m, s)
        lp = lp * active
    return lp.sum(dim=1)  # (b, s)


def _logJacobianZtoL(z: torch.Tensor, L: torch.Tensor, q: int) -> torch.Tensor:
    """Log |det J| of the z -> L map from unconstrainedToCholesky. Returns (...,)."""
    log_jac = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
    cursor = 0
    for i in range(1, q):
        w = torch.tanh(z[..., cursor : cursor + i])  # (..., i)
        # log(sech²(z)) = log(1 - tanh²(z))
        log_jac = log_jac + torch.log1p(-(w**2) + 1e-12).sum(-1)
        # remaining_{i,j} = 1 - sum_{k<j} L_{ik}^2; stored on L diagonal would need
        # recomputation - reconstruct from L off-diagonal entries for this row
        L_row = L[..., i, :i]  # (..., i)
        cum_sq = torch.nn.functional.pad(L_row.pow(2).cumsum(-1)[..., :-1], (1, 0))
        remaining = (1.0 - cum_sq).clamp(min=1e-8)  # (..., i)
        log_jac = log_jac + (0.5 * remaining.log()).sum(-1)
        cursor += i
    return log_jac


def logProbCorrRfx(
    z_corr: torch.Tensor,  # (b, s, d_corr)
    q: int,
    eta: torch.Tensor,  # (b,)  LKJ concentration; 0 = inactive (q < 2 for this dataset)
    q_active: torch.Tensor | None = None,  # (b,) per-dataset rfx dimension; None = q for all
) -> torch.Tensor:
    """Log prior for z_corr in unconstrained space. Returns (b, s).

    Datasets with eta=0 (q<2, no correlation) get zero contribution - their z_corr
    dimensions are masked in the flow and carry no prior information.

    With ``q_active`` the LKJ density is evaluated in each dataset's own dimension q_i
    (the leading q_i(q_i-1)/2 entries of z_corr are its active ones). Evaluating a
    q_i < q dataset in the padded q×q space is not equivalent: the LKJCholesky exponent
    of L_ii is 2(eta-1) + q - i, so padding adds (q - q_i) log L_ii per active row plus
    a different normalizer - a sample-dependent tilt of the prior toward zero correlation.
    """
    b, s = z_corr.shape[:2]
    lp = torch.zeros(b, s, dtype=z_corr.dtype, device=z_corr.device)
    active = eta > 0  # (b,) - datasets with an LKJ prior (q >= 2)
    if not active.any():
        return lp
    q_active = torch.full_like(eta, q, dtype=torch.long) if q_active is None else q_active.long()
    for q_i in torch.unique(q_active[active]).tolist():
        if q_i < 2:
            continue
        sel = active & (q_active == q_i)  # (b,)
        d_i = q_i * (q_i - 1) // 2
        z_a = z_corr[sel][..., :d_i]  # (b_a, s, d_i)
        L = unconstrainedToCholesky(z_a, q_i)  # (b_a, s, q_i, q_i)
        concentration = eta[sel].unsqueeze(-1).expand(-1, s)  # (b_a, s)
        lp_lkj = D.LKJCholesky(dim=q_i, concentration=concentration).log_prob(L)  # (b_a, s)
        lp[sel] = lp_lkj + _logJacobianZtoL(z_a, L, q_i)
    return lp


def _linearPredictor(
    ffx: torch.Tensor,  # (b, s, d)
    rfx: torch.Tensor,  # (b, m, s, q)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
) -> torch.Tensor:
    """Compute linear predictor eta = X @ ffx + Z * rfx. Returns (b, m, n, s)."""
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    return mu_g + mu_l


def _llNormal(
    eta: torch.Tensor,  # (b, m, n, s)
    sigma_eps: torch.Tensor,  # (b, s)
    y: torch.Tensor,  # (b, m, n, 1)
) -> torch.Tensor:
    scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
    return D.Normal(loc=eta, scale=scale).log_prob(y)


def _llBernoulli(
    eta: torch.Tensor,  # (b, m, n, s)
    sigma_eps: torch.Tensor,  # unused
    y: torch.Tensor,  # (b, m, n, 1)
) -> torch.Tensor:
    return D.Bernoulli(logits=eta).log_prob(y)


def _llPoisson(
    eta: torch.Tensor,  # (b, m, n, s)
    sigma_eps: torch.Tensor,  # unused
    y: torch.Tensor,  # (b, m, n, 1)
) -> torch.Tensor:
    frac = float((eta > POISSON_ETA_CLIP_MAX).float().mean().item())
    _warnIfFrequentClipping('poisson_eta_torch_ll', frac)
    rate = torch.exp(eta.clamp(max=POISSON_ETA_CLIP_MAX))
    return D.Poisson(rate=rate).log_prob(y)


_LIKELIHOOD_LL = (_llNormal, _llBernoulli, _llPoisson)


def _rfxPrecision(
    sigma_rfx: torch.Tensor,  # (b, s, q)
    sigma_eps: torch.Tensor,  # (b, s)
    ZtZ: torch.Tensor,  # (b, m, q, q)
    L_corr: torch.Tensor | None,  # (b, s, q, q)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rfx posterior precision of a Normal model, shared by the marginal and both
    conditionals so that weights and draws use one factor.

    Returns the Cholesky factor of the σ²_eps-scaled precision M = σ²_eps Σ_rfx⁻¹ + ZᵀZ per group
    (b, m, s, q, q), log det Σ_rfx (b, s) and the degenerate states (b, m, s). Unlike
    Σ_rfx⁻¹ + ZᵀZ / σ²_eps, M stays finite as σ_eps -> 0 (#57). Its jitter σ²_eps 1e-6 I is 1e-6
    on the unscaled precision. A state is degenerate where M is not finite or does not factor:
    it has zero posterior mass, and its factor is the identity so that solves stay finite.

    Σ_rfx is diag(σ_rfx²), or D L_corr L_corrᵀ D with D = diag(σ_rfx) when L_corr is given;
    σ_rfx is clamped at 1e-6, since the flow can underflow to 0.
    """
    sigma_rfx = sigma_rfx.clamp(min=1e-6)
    q = sigma_rfx.shape[-1]
    eye = torch.eye(q, dtype=sigma_rfx.dtype, device=sigma_rfx.device)
    if L_corr is None:
        Sigma_inv = torch.diag_embed(1.0 / sigma_rfx.pow(2))  # (b, s, q, q)
        log_det_Sigma = 2.0 * sigma_rfx.log().sum(-1)  # (b, s)
    else:
        L_Sigma = sigma_rfx.unsqueeze(-1) * L_corr  # (b, s, q, q): D @ L_corr
        Sigma_inv = choleskySolve(eye.expand_as(L_Sigma), L_Sigma)
        diag_corr = L_corr.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
        log_det_Sigma = 2.0 * (sigma_rfx.log() + diag_corr.log()).sum(-1)
    s2e = sigma_eps.pow(2)[:, None, :, None, None]  # (b, 1, s, 1, 1)
    M = s2e * (Sigma_inv.unsqueeze(1) + 1e-6 * eye) + ZtZ.unsqueeze(2)  # (b, m, s, q, q)
    chol, info = torch.linalg.cholesky_ex(M)
    degenerate = (info != 0) | ~torch.isfinite(M).all(-1).all(-1)  # (b, m, s)
    chol = torch.where(degenerate[..., None, None], eye, chol)
    return chol, log_det_Sigma, degenerate


def logMarginalLikelihoodNormal(
    ffx: torch.Tensor,  # (b, s, d)
    sigma_rfx: torch.Tensor,  # (b, s, q)
    sigma_eps: torch.Tensor,  # (b, s)
    y: torch.Tensor,  # (b, m, n, 1)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    mask_n: torch.Tensor,  # (b, m, n, 1)
    mask_m: torch.Tensor,  # (b, m, 1)
    L_corr: torch.Tensor | None = None,  # (b, s, q, q) Cholesky of the rfx correlation matrix
) -> torch.Tensor:
    """Marginal log-likelihood for Normal, with rfx integrated out analytically.

    Exploits the Woodbury identity to work with a q×q system per group instead
    of the n_i×n_i marginal covariance directly:

        V_i = Z_i Σ_rfx Z_i^T + σ²_eps I_{n_i}
        M_i = σ²_eps Σ_rfx^{-1} + Z_i^T Z_i             (q×q, _rfxPrecision)
        log det V_i = log det M_i + log det Σ_rfx + (n_i − q) log σ²_eps
        r_i^T V_i^{-1} r_i = (‖r_i‖² − h_i^T M_i^{-1} h_i) / σ²_eps

    where r_i = y_i − X_i β and h_i = Z_i^T r_i. Degenerate states get −inf.

    Σ_rfx is diag(σ_rfx²) by default; pass L_corr (from unconstrainedToCholesky)
    for the correlated case Σ_rfx = D L_corr L_corrᵀ D with D = diag(σ_rfx).
    Omitting L_corr for correlated-rfx samples silently targets the wrong
    (independent-rfx) marginal - the cause of the Corr(RFX) degradation seen in
    the 2026-07 posthoc ablation.

    Padded q-dimensions (inactive rfx) contribute zero net log-det because the
    Z_i column is zero, making the padded diagonal of M_i equal to σ²_eps/σ²_rfx,
    which exactly cancels the corresponding terms in log det Σ_rfx and q log σ²_eps. This holds
    in the correlated case too: masked z_corr entries make the padded rows of
    L_corr identity rows.

    Returns (b, s).
    """
    # precompute ZtZ - independent of samples (b, m, q, q)
    Z_m = Z * mask_n  # zero out padded observations
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)

    # residuals and group-level sufficient statistics
    mu = torch.einsum('bmnd,bsd->bmns', X, ffx)   # (b, m, n, s)
    r = (y - mu) * mask_n                          # (b, m, n, s)
    rtr = r.pow(2).sum(dim=2)                      # (b, m, s): ‖r_i‖²
    h = torch.einsum('bmnq,bmns->bmsq', Z_m, r)   # (b, m, s, q): Z_i^T r_i

    # valid observations per group
    n_i = mask_n.squeeze(-1).sum(dim=-1).float()   # (b, m)

    chol_M, log_det_Sigma, degenerate = _rfxPrecision(sigma_rfx, sigma_eps, ZtZ, L_corr)
    log_det_M = 2.0 * chol_M.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (b, m, s)

    # h^T M^{-1} h via Cholesky solve
    M_inv_h = choleskySolve(h.unsqueeze(-1), chol_M).squeeze(-1)  # (b, m, s, q)
    hMh = (h * M_inv_h).sum(-1)                                    # (b, m, s)

    # log det V_i (padded dims of log_det_Sigma cancel with M's padded diagonal)
    s2e = sigma_eps.pow(2)[:, None, :]               # (b, 1, s)
    n_free = n_i[:, :, None] - Z.shape[-1]           # (b, m, 1)
    log_det_V = log_det_M + log_det_Sigma[:, None, :] + n_free * s2e.log()  # (b, m, s)

    # per-group log-likelihood
    quad = (rtr - hMh) / s2e                        # (b, m, s)
    ll_i = -0.5 * (n_i[:, :, None] * math.log(2.0 * math.pi) + log_det_V + quad)
    ll_i = torch.where(degenerate, -math.inf, ll_i)

    # sum over valid groups
    return torch.where(mask_m.bool(), ll_i, 0.0).sum(dim=1)  # (b, s)


def sampleRfxConditionalNormal(
    ffx: torch.Tensor,  # (b, s, d)
    sigma_rfx: torch.Tensor,  # (b, s, q)
    sigma_eps: torch.Tensor,  # (b, s)
    y: torch.Tensor,  # (b, m, n, 1)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    mask_n: torch.Tensor,  # (b, m, n, 1)
    mask_m: torch.Tensor,  # (b, m, 1)
    L_corr: torch.Tensor | None = None,  # (b, s, q, q) Cholesky of the rfx correlation matrix
) -> torch.Tensor:
    """Draw rfx ~ p(rfx | θ_g, y) using Normal-Normal conjugacy.

    For each global sample and each group j:
        V_j⁻¹ = Σ_rfx⁻¹ + Zⱼᵀ Zⱼ / σ²_eps
        μ_j   = V_j (Zⱼᵀ rⱼ / σ²_eps)     where rⱼ = yⱼ − Xⱼ β
        rfx_j ~ N(μ_j, V_j)

    Σ_rfx = diag(σ_rfx²) by default, or D L_corr L_corrᵀ D when L_corr is given.
    This is the exact conditional dual to logMarginalLikelihoodNormal - pass the
    same L_corr to both so weights and rfx draws share one target (the mismatch
    documented in posthoc/metropolis.py's Findings section).

    Returns (b, m, s, q).
    """
    # Residuals: r = y - X @ ffx  (b, m, n, s)
    mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)
    r = (y - mu_ffx) * mask_n

    # Group-level sufficient statistics
    Z_m = Z * mask_n  # zero out padded observations
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)  # (b, m, q, q)
    Ztr = torch.einsum('bmnq,bmns->bmsq', Z_m, r)    # (b, m, s, q)

    # M = σ²_eps V⁻¹ (b, m, s, q, q): unlike V⁻¹ itself it stays finite as σ_eps -> 0, where
    # Zᵀr / σ²_eps overflows float32 although the conditional tends to the least-squares fit (#57:
    # an IMH chain stuck on a flow draw with σ_eps 2e-19). Degenerate states carry zero weight in
    # logMarginalLikelihoodNormal, which fails the same factor, so their zero draw never enters a
    # result.
    chol_M, _, degenerate = _rfxPrecision(sigma_rfx, sigma_eps, ZtZ, L_corr)

    # Posterior mean: μ = V (Zᵀr / σ²_eps) = M⁻¹ Zᵀr
    post_mean = choleskySolve(Ztr.unsqueeze(-1), chol_M).squeeze(-1)  # (b, m, s, q)

    # Sample: rfx = μ + σ_eps chol_M⁻ᵀ z  where z ~ N(0, I)
    # chol_M is Chol(M) and V = σ²_eps M⁻¹, so Chol(V) = σ_eps chol_M⁻ᵀ (upper-triangular solve)
    z = torch.randn_like(post_mean).unsqueeze(-1)  # (b, m, s, q, 1)
    unit = torch.linalg.solve_triangular(chol_M.mT, z, upper=True).squeeze(-1)  # (b, m, s, q)
    rfx = post_mean + sigma_eps[:, None, :, None] * unit  # (b, m, s, q)
    rfx = torch.where(degenerate[..., None], 0.0, rfx)

    return rfx * mask_m.unsqueeze(-1).float()  # (b, m, s, q)


def sampleFfxConditionalNormal(
    ffx: torch.Tensor,  # (b, s, d) current draw: kept on padded dims and on rejection
    sigma_rfx: torch.Tensor,  # (b, s, q)
    sigma_eps: torch.Tensor,  # (b, s)
    y: torch.Tensor,  # (b, m, n, 1)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    mask_n: torch.Tensor,  # (b, m, n, 1)
    mask_m: torch.Tensor,  # (b, m, 1)
    nu_ffx: torch.Tensor,  # (b, 1, d) prior location
    tau_ffx: torch.Tensor,  # (b, 1, d) prior scale
    family_ffx: torch.Tensor,  # (b,) index into FFX_FAMILIES
    mask_d: torch.Tensor,  # (b, 1, d)
    L_corr: torch.Tensor | None = None,  # (b, s, q, q) Cholesky of the rfx correlation matrix
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ffx ~ p(ffx | σ_rfx, σ_eps, y) with the rfx integrated out.

    Under the normal prior N(ν, τ²) the conditional is Gaussian (generalized least squares on
    the V_j of logMarginalLikelihoodNormal, Woodbury per group):

        P = Σ_j X_jᵀ V_j⁻¹ X_j + diag(1/τ²),   μ = P⁻¹ (Σ_j X_jᵀ V_j⁻¹ y_j + ν/τ²),
        X_jᵀ V_j⁻¹ A = (X_jᵀ A − X_jᵀ Z_j M_j⁻¹ Z_jᵀ A) / σ²_eps,   M_j = σ²_eps Σ_rfx⁻¹ + Z_jᵀ Z_j.

    Under the Student-t prior that Gaussian serves as an independence-MH proposal for the exact
    conditional, whose ratio reduces to the prior ratio p_t(ffx*) p_N(ffx) / (p_t(ffx) p_N(ffx*)).
    At σ_eps near 0 the Woodbury terms cancel and rounding can leave P indefinite; such states
    and degenerate ones (_rfxPrecision) keep the current draw, as on rejection (#57). Computed in
    float64. Returns (ffx (b, s, d), accepted (b, s) bool).
    """
    dtype = torch.float64
    b, s, d = ffx.shape
    q = Z.shape[-1]
    mn = mask_n.to(dtype)
    X_m, Z_m, y_m = X.to(dtype) * mn, Z.to(dtype) * mn, y.to(dtype) * mn
    XtX, XtZ, Xty = (
        X_m.mT @ X_m,
        X_m.mT @ Z_m,
        X_m.mT @ y_m,
    )  # (b, m, d, d), (b, m, d, q), (b, m, d, 1)
    ZtZ, Zty = Z_m.mT @ Z_m, Z_m.mT @ y_m  # (b, m, q, q), (b, m, q, 1)
    m = X.shape[1]

    L = None if L_corr is None else L_corr.to(dtype)
    chol_M, _, degenerate = _rfxPrecision(sigma_rfx.to(dtype), sigma_eps.to(dtype), ZtZ, L)
    inv_s2e = (1.0 / sigma_eps.to(dtype).pow(2))[:, None, :, None, None]  # (b, 1, s, 1, 1)
    Minv_ZtX = choleskySolve(XtZ.mT[:, :, None].expand(b, m, s, q, d), chol_M)  # (b, m, s, q, d)
    Minv_Zty = choleskySolve(Zty[:, :, None].expand(b, m, s, q, 1), chol_M)  # (b, m, s, q, 1)
    G = (XtX[:, :, None] - XtZ[:, :, None] @ Minv_ZtX) * inv_s2e  # (b, m, s, d, d)
    g = (Xty[:, :, None] - XtZ[:, :, None] @ Minv_Zty) * inv_s2e  # (b, m, s, d, 1)
    mm = mask_m.to(dtype)[:, :, None, :, None]  # (b, m, 1, 1, 1)
    G, g = (G * mm).sum(1), (g * mm).sum(1).squeeze(-1)  # (b, s, d, d), (b, s, d)

    # padded dims: unit precision, zero mean, then the input value is kept
    active = mask_d.bool()  # (b, 1, d)
    nu, tau = nu_ffx.to(dtype), tau_ffx.to(dtype)
    prec = torch.where(active, 1.0 / tau.pow(2), 1.0)  # (b, 1, d)
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)  # (b, 1, d, d)
    P = G * pair + torch.diag_embed(prec.expand(b, s, d))
    L_P, info_P = torch.linalg.cholesky_ex(P)
    factored = (info_P == 0) & ~degenerate.any(1)  # (b, s)
    L_P = torch.where(factored[..., None, None], L_P, torch.eye(d, dtype=dtype, device=X.device))
    mean = choleskySolve(((g + nu * prec) * active).unsqueeze(-1), L_P).squeeze(-1)
    z = torch.randn_like(mean)
    draw = mean + torch.linalg.solve_triangular(L_P.mT, z.unsqueeze(-1), upper=True).squeeze(-1)

    cur = ffx.to(dtype)
    normal = torch.zeros_like(family_ffx)
    log_ratio = (
        logProbFfx(draw, nu, tau, family_ffx, active)
        - logProbFfx(draw, nu, tau, normal, active)
        - logProbFfx(cur, nu, tau, family_ffx, active)
        + logProbFfx(cur, nu, tau, normal, active)
    )
    # (b, s); 0 < 0 never: the normal prior always accepts where P factored
    accepted = (torch.rand_like(log_ratio).log() < log_ratio) & factored
    draw = torch.where(active & accepted[..., None], draw, cur)
    return draw.to(ffx.dtype), accepted


def logLikelihood(
    ffx: torch.Tensor,  # (b, s, d)
    sigma_eps: torch.Tensor,  # (b, s)
    rfx: torch.Tensor,  # (b, m, s, q)
    y: torch.Tensor,  # (b, m, n, 1)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    mask: torch.Tensor | None = None,  # (b, m, n, 1)
    likelihood_family: int = 0,
) -> torch.Tensor:
    """Conditional log-likelihood dispatched by likelihood family."""
    eta = _linearPredictor(ffx, rfx, X, Z)
    ll = _LIKELIHOOD_LL[likelihood_family](eta, sigma_eps, y)
    if mask is not None:
        ll = ll * mask
    return ll.sum(dim=(1, 2))  # (b, s)


# ---------------------------------------------------------------------------
# Torch posterior predictive distribution
# ---------------------------------------------------------------------------


def posteriorPredictiveDist(
    ffx: torch.Tensor,  # (b, s, d)
    sigma_eps: torch.Tensor,  # (b, s)
    rfx: torch.Tensor,  # (b, m, s, q)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    likelihood_family: int = 0,
) -> D.Distribution:
    """Return posterior predictive distribution p(y | theta) for the given likelihood."""
    eta = _linearPredictor(ffx, rfx, X, Z)  # (b, m, n, s)
    if likelihood_family == 0:  # normal
        scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
        return D.Normal(loc=eta, scale=scale)
    elif likelihood_family == 1:  # bernoulli
        return D.Bernoulli(logits=eta)
    elif likelihood_family == 2:  # poisson
        frac = float((eta > POISSON_ETA_CLIP_MAX).float().mean().item())
        _warnIfFrequentClipping('poisson_eta_torch_ppd', frac)
        rate = torch.exp(eta.clamp(max=POISSON_ETA_CLIP_MAX))
        return D.Poisson(rate=rate)
    raise ValueError(f'unknown likelihood family: {likelihood_family}')


# ---------------------------------------------------------------------------
# Encoding for neural network context
# ---------------------------------------------------------------------------


def oneHotFamily(family: torch.Tensor, n_families: int) -> torch.Tensor:
    """One-hot encode family indices. family: (b,) -> (b, n_families)."""
    return torch.nn.functional.one_hot(family.long(), n_families).float()


class FamilyEncoder(torch.nn.Module):
    """Encode family indices as either one-hot or learned embeddings.

    Args:
        n_families: number of families for each parameter group
        embed_dim: if None, use one-hot encoding; otherwise learned embedding of this dim
    """

    def __init__(
        self,
        n_families: tuple[int, ...],
        embed_dim: int | None = None,
    ):
        super().__init__()
        self.n_families = n_families
        self.embed_dim = embed_dim
        if embed_dim is not None:
            self.embeddings = torch.nn.ModuleList(
                [torch.nn.Embedding(n, embed_dim) for n in n_families]
            )
        else:
            self.embeddings = None

    @property
    def d_output(self) -> int:
        if self.embed_dim is not None:
            return self.embed_dim * len(self.n_families)
        return sum(self.n_families)

    def forward(self, families: list[torch.Tensor]) -> torch.Tensor:
        """Encode a list of family index tensors, each (b,). Returns (b, d_output)."""
        if self.embeddings is not None:
            parts = [emb(f.long()) for emb, f in zip(self.embeddings, families)]
        else:
            parts = [oneHotFamily(f, n) for f, n in zip(families, self.n_families)]
        return torch.cat(parts, dim=-1)
