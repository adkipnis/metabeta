import math
import numpy as np
import torch
from scipy.stats import norm, t
from torch import distributions as D


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFX_FAMILIES = ('normal', 'student')
SIGMA_FAMILIES = ('halfnormal', 'halfstudent')

STUDENT_DF = 4


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


_FFX_SAMPLE_NP = (_sampleNormalNp, _sampleStudentNp)
_SIGMA_SAMPLE_NP = (_sampleHalfNormalNp, _sampleHalfStudentNp)


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


_FFX_LOG_PROB = (_logProbNormal, _logProbStudent)
_SIGMA_LOG_PROB = (_logProbHalfNormal, _logProbHalfStudent)


def logProbFfx(
    x: torch.Tensor,       # (b, s, d)
    loc: torch.Tensor,     # (b, 1, d)
    scale: torch.Tensor,   # (b, 1, d)
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
    x: torch.Tensor,       # (b, s, q) or (b, s)
    scale: torch.Tensor,   # (b, 1, q) or (b, 1)
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
