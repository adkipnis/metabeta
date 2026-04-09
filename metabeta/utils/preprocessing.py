import torch
import numpy as np
from dataclasses import dataclass, field


# --- checkers
def checkBinary(x: np.ndarray, axis: int = 0) -> np.ndarray:
    binary = (x == 0) + (x == 1)
    return np.all(binary, axis=axis)


def checkContinuous(x: np.ndarray, axis: int = 0, tol: float = 1e-12) -> np.ndarray:
    diffs = np.abs(x - x.round())
    too_large = diffs > tol
    return np.all(too_large, axis=axis)


def checkConstant(x: np.ndarray, axis: int = 0) -> np.ndarray:
    same = x[0, :] == x
    return np.all(same, axis=axis)


def checkCountLike(x: np.ndarray, axis: int = 0, tol: float = 1e-12) -> np.ndarray:
    is_int = np.all(np.abs(x - x.round()) <= tol, axis=axis)
    is_nonneg = np.all(x >= -tol, axis=axis)
    is_binary = checkBinary(x, axis=axis)
    return is_int & is_nonneg & (~is_binary)


# --- standardize
def moments(
    x: np.ndarray,
    axis: int = 0,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis, keepdims=True)
    std = x.std(axis, keepdims=True)
    if exclude is not None:
        exclude = exclude.reshape(mean.shape)
        mean[exclude] = 0
        std[exclude] = 1
    return mean, std


def standardize(
    x: np.ndarray, axis: int = 0, exclude_binary: bool = True, eps: float = 1e-6
) -> np.ndarray:
    exclude = checkBinary(x, axis=axis) if exclude_binary else None
    mean, std = moments(x, axis, exclude=exclude)
    bad = (~np.isfinite(std)) | (std < eps)
    std = np.where(bad, 1.0, std)
    return (x - mean) / std


def transformPredictors(
    x: np.ndarray,
    axis: int = 0,
    exclude_binary: bool = True,
    transform_counts: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f'expected 2D predictors, got shape={x.shape}')
    if axis != 0:
        raise ValueError('transformPredictors currently supports axis=0 only')

    out = x.astype(float, copy=True)
    if transform_counts:
        count_like = checkCountLike(out, axis=axis)
        if np.any(count_like):
            out[:, count_like] = np.log1p(out[:, count_like])

    return standardize(out, axis=axis, exclude_binary=exclude_binary, eps=eps)


@dataclass
class Standardizer:
    axis: int = 0
    exclude_binary: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        exclude = checkBinary(x, axis=self.axis) if self.exclude_binary else None
        self.mean, self.std = moments(x, self.axis, exclude=exclude)
        return (x - self.mean) / self.std

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class NumericTransformer:
    """Stateful fit/transform for numeric predictors.

    Column classification:
      - binary {0,1}: pass-through unchanged.
      - count-like with |skew| > log1p_skew_threshold: log1p then z-standardise.
      - count-like with low skew: z-standardise only (log1p would impose an
        unwarranted multiplicative functional form on a near-symmetric variable).
      - continuous: z-standardise.
    """

    exclude_binary: bool = True
    transform_counts: bool = True
    log1p_skew_threshold: float = 1.0  # only log1p count-like cols with |skew| above this
    eps: float = 1e-6
    # fitted state
    is_binary_: np.ndarray | None = field(default=None, repr=False)
    is_count_like_: np.ndarray | None = field(default=None, repr=False)
    is_log1p_: np.ndarray | None = field(default=None, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, x: np.ndarray) -> 'NumericTransformer':
        if x.ndim != 2:
            raise ValueError(f'expected 2D array, got shape={x.shape}')
        x = x.astype(float)
        n, d = x.shape

        self.is_binary_ = checkBinary(x, axis=0) if self.exclude_binary else np.zeros(d, dtype=bool)
        self.is_count_like_ = (
            checkCountLike(x, axis=0) & ~self.is_binary_
            if self.transform_counts
            else np.zeros(d, dtype=bool)
        )

        # log1p only for count-like columns that are sufficiently right-skewed;
        # mildly skewed count predictors are z-standardised on the raw scale to
        # avoid changing the functional form of the relationship.
        self.is_log1p_ = np.zeros(d, dtype=bool)
        if self.is_count_like_.any() and n > 2:
            from scipy.stats import skew as _skew

            skewness = np.zeros(d)
            skewness[self.is_count_like_] = _skew(x[:, self.is_count_like_], axis=0)
            self.is_log1p_ = self.is_count_like_ & (np.abs(skewness) > self.log1p_skew_threshold)

        # apply log1p to the selected subset of count-like columns;
        # clip to [0, inf) first as a safeguard even though checkCountLike already
        # enforces non-negativity — log1p is undefined for negative inputs.
        x_work = x.copy()
        if self.is_log1p_.any():
            x_work[:, self.is_log1p_] = np.clip(x_work[:, self.is_log1p_], 0, None)
            x_work[:, self.is_log1p_] = np.log1p(x_work[:, self.is_log1p_])

        # z-standardise all non-binary columns (log1p-transformed + raw count-like + continuous)
        needs_z = ~self.is_binary_
        self.mean_ = np.zeros(d)
        self.std_ = np.ones(d)
        if needs_z.any():
            self.mean_[needs_z] = x_work[:, needs_z].mean(axis=0)
            std = x_work[:, needs_z].std(axis=0)
            bad = (~np.isfinite(std)) | (std < self.eps)
            self.std_[needs_z] = np.where(bad, 1.0, std)

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
def rescaleData(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    data = {k: v.clone() for k, v in data.items()}  # avoids side effects
    for key in (
        'y',
        # params
        'ffx',
        'sigma_rfx',
        'sigma_eps',
        'rfx',
        # hyperparams
        'nu_ffx',
        'tau_ffx',
        'tau_rfx',
        'tau_eps',
    ):
        if key not in data:
            continue
        sd_y = data['sd_y'].view(-1, *([1] * (data[key].ndim - 1)))
        data[key] *= sd_y
    return data
