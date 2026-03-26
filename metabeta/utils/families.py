import math
import numpy as np
import torch
from scipy.stats import expon, norm, t
from torch import distributions as D


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFX_FAMILIES = ('normal', 'student')
SIGMA_FAMILIES = ('halfnormal', 'halfstudent', 'exponential')

FFX_FAMILY_PROBS = (0.80, 0.20)
SIGMA_RFX_FAMILY_PROBS = (0.60, 0.30, 0.10)
SIGMA_EPS_FAMILY_PROBS = (0.50, 0.40, 0.10)

STUDENT_DF = 5

LIKELIHOOD_FAMILIES = ('normal', 'bernoulli')
LIKELIHOOD_HAS_SIGMA_EPS = (True, False)


def hasSigmaEps(likelihood: int) -> bool:
    """Whether the likelihood family has a residual variance parameter."""
    return LIKELIHOOD_HAS_SIGMA_EPS[likelihood]


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
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def simulateNormalNp(
    rng: np.random.Generator,
    eta: np.ndarray,
    sigma_eps: float,
) -> np.ndarray:
    eps = rng.normal(size=eta.shape)
    return eta + eps * sigma_eps


def simulateBernoulliNp(
    rng: np.random.Generator,
    eta: np.ndarray,
    sigma_eps: float,
) -> np.ndarray:
    p = _expit(eta)
    return rng.binomial(1, p).astype(eta.dtype)


_LIKELIHOOD_SIMULATE_NP = (simulateNormalNp, simulateBernoulliNp)


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


_LIKELIHOOD_LL = (_llNormal, _llBernoulli)


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
