import math
import logging
import numpy as np
import torch
from scipy.stats import expon, norm, t
from torch import distributions as D

from metabeta.utils.preprocessing import standardize
from metabeta.utils.regularization import unconstrainedToCholeskyCorr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFX_FAMILIES = ('normal', 'student')
SIGMA_FAMILIES = ('halfnormal', 'halfstudent', 'exponential')

FFX_FAMILY_PROBS = (0.80, 0.20)
SIGMA_RFX_FAMILY_PROBS = (0.60, 0.30, 0.10)
SIGMA_EPS_FAMILY_PROBS = (0.50, 0.40, 0.10)

STUDENT_DF = 5

LIKELIHOOD_FAMILIES = ('normal', 'bernoulli', 'poisson')
LIKELIHOOD_HAS_SIGMA_EPS = (True, False, False)
LIKELIHOOD_BAMBI_FAMILY = ('gaussian', 'bernoulli', 'poisson')

# numerical stabilization constants
BERNOULLI_LOGIT_CLIP_ABS = 500.0
BERNOULLI_ETA_ABS_MAX = 10.0
BERNOULLI_REROLL_EXTREME_FRACTION_MAX = 0.05
BERNOULLI_REROLL_MAX_ATTEMPTS = 20
POISSON_ETA_CLIP_MAX = 10.0
POISSON_X_CLIP_ABS = 5.0
POISSON_REROLL_CLIP_FRACTION_MAX = 0.01
POISSON_REROLL_MAX_ATTEMPTS = 20
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


def hasSigmaEps(likelihood: int) -> bool:
    """Whether the likelihood family has a residual variance parameter."""
    return LIKELIHOOD_HAS_SIGMA_EPS[likelihood]


def bambiFamilyName(likelihood: int) -> str:
    """Bambi family string for the given likelihood index."""
    return LIKELIHOOD_BAMBI_FAMILY[likelihood]


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
# Torch sampling (prior predictive) — select approach
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
    eps = rng.normal(size=eta.shape)
    eps = standardize(eps, axis=0) * sigma_eps
    return eta + eps


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
    lp = dist.log_prob(rfx)  # rfx (b, m, s, q) → lp (b, m, s)
    if mask is not None:
        # mask shape (b, m, 1, q) — reduce to (b, m, s) by checking any active q dim
        active = mask[..., 0].expand_as(lp)  # (b, m, s)
        lp = lp * active
    return lp.sum(dim=1)  # (b, s)


def _logJacobianZtoL(z: torch.Tensor, L: torch.Tensor, q: int) -> torch.Tensor:
    """Log |det J| of the z → L map from unconstrainedToCholeskyCorr. Returns (...,)."""
    log_jac = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
    cursor = 0
    for i in range(1, q):
        w = torch.tanh(z[..., cursor : cursor + i])  # (..., i)
        # log(sech²(z)) = log(1 - tanh²(z))
        log_jac = log_jac + torch.log1p(-(w ** 2) + 1e-12).sum(-1)
        # remaining_{i,j} = 1 - sum_{k<j} L_{ik}^2; stored on L diagonal would need
        # recomputation — reconstruct from L off-diagonal entries for this row
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
) -> torch.Tensor:
    """Log prior for z_corr in unconstrained space. Returns (b, s).

    Datasets with eta=0 (q<2, no correlation) get zero contribution — their z_corr
    dimensions are masked in the flow and carry no prior information.
    """
    b, s = z_corr.shape[:2]
    lp = torch.zeros(b, s, dtype=z_corr.dtype, device=z_corr.device)
    active = eta > 0  # (b,) — datasets with an LKJ prior (q >= 2)
    if not active.any():
        return lp
    z_a = z_corr[active]  # (b_a, s, d_corr)
    L = unconstrainedToCholeskyCorr(z_a, q)  # (b_a, s, q, q)
    concentration = eta[active].unsqueeze(-1).expand(-1, s)  # (b_a, s)
    lp_lkj = D.LKJCholesky(dim=q, concentration=concentration).log_prob(L)  # (b_a, s)
    log_jac = _logJacobianZtoL(z_a, L, q)  # (b_a, s)
    lp[active] = lp_lkj + log_jac
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


def logMarginalLikelihoodNormal(
    ffx: torch.Tensor,  # (b, s, d)
    sigma_rfx: torch.Tensor,  # (b, s, q)
    sigma_eps: torch.Tensor,  # (b, s)
    y: torch.Tensor,  # (b, m, n, 1)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    mask_n: torch.Tensor,  # (b, m, n, 1)
    mask_m: torch.Tensor,  # (b, m, 1)
) -> torch.Tensor:
    """Marginal log-likelihood for Normal, with rfx integrated out analytically.

    Exploits the Woodbury identity to work with a q×q system per group instead
    of the n_i×n_i marginal covariance directly:

        V_i = Z_i Σ_rfx Z_i^T + σ²_eps I_{n_i}
        M_i = Σ_rfx^{-1} + Z_i^T Z_i / σ²_eps           (q×q)
        log det V_i = log det M_i + log det Σ_rfx + n_i log σ²_eps
        r_i^T V_i^{-1} r_i = (‖r_i‖² − h_i^T M_i^{-1} h_i / σ²_eps) / σ²_eps

    where r_i = y_i − X_i β and h_i = Z_i^T r_i.

    Padded q-dimensions (inactive rfx) contribute zero net log-det because the
    Z_i column is zero, making the padded diagonal of M_i equal to 1/σ²_rfx,
    which exactly cancels the corresponding term in log det Σ_rfx.

    Returns (b, s).
    """
    # precompute ZtZ — independent of samples (b, m, q, q)
    Z_m = Z * mask_n  # zero out padded observations
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)

    # residuals and group-level sufficient statistics
    mu = torch.einsum('bmnd,bsd->bmns', X, ffx)   # (b, m, n, s)
    r = (y - mu) * mask_n                          # (b, m, n, s)
    rtr = r.pow(2).sum(dim=2)                      # (b, m, s): ‖r_i‖²
    h = torch.einsum('bmnq,bmns->bmsq', Z_m, r)   # (b, m, s, q): Z_i^T r_i

    # valid observations per group
    n_i = mask_n.squeeze(-1).sum(dim=-1).float()   # (b, m)

    # clamp sigma_rfx away from zero: flow can underflow to 0, which makes
    # both Sigma_inv and log(sigma_rfx) numerically degenerate.
    sigma_rfx = sigma_rfx.clamp(min=1e-6)

    # M_i = Σ_rfx^{-1} + Z_i^T Z_i / σ²_eps  →  (b, m, s, q, q)
    s2e = sigma_eps.pow(2)                                          # (b, s)
    s2r_inv = 1.0 / sigma_rfx.pow(2)                               # (b, s, q)
    Sigma_inv = torch.diag_embed(s2r_inv).unsqueeze(1)             # (b, 1, s, q, q)
    M = Sigma_inv + ZtZ.unsqueeze(2) / s2e[:, None, :, None, None]  # (b, m, s, q, q)

    # Cholesky factorisation of M — jitter diagonal for numerical stability.
    # Use cholesky_ex to avoid hard failures; bad elements get -inf log-likelihood.
    jitter = 1e-6 * torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    chol_M, info = torch.linalg.cholesky_ex(M + jitter)            # (b, m, s, q, q)
    chol_ok = info == 0                                           # (b, m, s)

    log_det_M = 2.0 * chol_M.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (b, m, s)
    log_det_M = torch.where(chol_ok, log_det_M, log_det_M.new_tensor(torch.inf))

    # h^T M^{-1} h via Cholesky solve
    M_inv_h = torch.cholesky_solve(h.unsqueeze(-1), chol_M).squeeze(-1)  # (b, m, s, q)
    hMh = (h * M_inv_h).sum(-1)                                    # (b, m, s)

    # log det V_i = log det M_i + log det Σ_rfx + n_i log σ²_eps
    log_det_Sigma = 2.0 * sigma_rfx.log().sum(-1)   # (b, s); padded dims cancel with M
    log_s2e = s2e.log()                              # (b, s)
    log_det_V = (
        log_det_M + log_det_Sigma[:, None, :] + n_i[:, :, None] * log_s2e[:, None, :]
    )  # (b, m, s)

    # quadratic form
    s2e_bms = s2e[:, None, :]                        # (b, 1, s)
    quad = (rtr - hMh / s2e_bms) / s2e_bms          # (b, m, s)

    # per-group log-likelihood
    ll_i = -0.5 * (n_i[:, :, None] * math.log(2.0 * math.pi) + log_det_V + quad)

    # sum over valid groups
    return (ll_i * mask_m).sum(dim=1)  # (b, s)


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
