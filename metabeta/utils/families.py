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
