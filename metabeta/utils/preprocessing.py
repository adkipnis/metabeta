import torch
import numpy as np
from dataclasses import dataclass

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


def rescaleData(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    data = {k: v.clone() for k, v in data.items()}
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
        sd_y = data['sd_y'].view(-1, *([1] * (data[key].ndim - 1)))
        data[key] *= sd_y
    return data
