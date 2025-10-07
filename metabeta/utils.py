import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------
# name utils
def getConsoleWidth() -> int:
    """Determine the width of the console for formatting purposes."""
    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        print("Could not determine console width. Defaulting to 80.")
        console_width = 80
    return console_width


def parseSize(size: int) -> str:
    if size >= 1e6:
        n = f"{size / 1e6:.0f}m"
    elif size >= 1e3:
        n = f"{size / 1e3:.0f}k"
    else:
        n = str(size)
    return n


# -----------------------------------------------------------------------------
# loading utils


def setDevice(device: str):
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def dsFilename(
    fx_type: str,
    ds_type: str,
    b: int,
    m: int,
    n: int,
    d: int,
    q: int,
    size: int,
    part: int = 0,
    ftype: str = "npz",
    varied: bool = False,
    tag: str = "",
    outside: bool = False,
) -> Path:
    """example: ffx-train-d=8-m=1-n=50-50k-001.npz"""
    s = parseSize(size)
    if "test" in ds_type:
        part = -1
    p = f"{part:03d}"
    if tag:
        tag = "-" + tag
    if varied:
        varied_ = "-varied"
    else:
        varied_ = ""

    first = ""
    if outside:
        first = ".."
    return Path(
        first,
        "outputs",
        "data",
        f"{fx_type}-{ds_type}-b={b}-m={m}-n={n}-d={d}-q={q}-{s}-{p}{varied_}{tag}.{ftype}",
    )


# -----------------------------------------------------------------------------
# size utils
def dInput(d_data: int, fx_type: str) -> int:
    n_fx = 2 if fx_type == "mfx" else 1
    return 1 + n_fx * (d_data - 1)


def nParams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# moment utils
def maskedMean(x: torch.Tensor, dim: tuple | int, mask: torch.Tensor) -> torch.Tensor:
    sums = x.sum(dim, keepdim=True)
    count = mask.sum(dim, keepdim=True)
    return sums / (count + 1e-12)


def maskedStd(
    x: torch.Tensor, dim: tuple, mask: torch.Tensor, mean: torch.Tensor | None = None
) -> torch.Tensor:
    if mean is None:
        mean = maskedMean(x, dim, mask)
    diff_squared_sum = ((x - mean) * mask).square().sum(dim, keepdim=True)
    count = mask.sum(dim, keepdim=True) - 1
    count = torch.where(count < 1, 1, count)
    return (diff_squared_sum / count).sqrt()


def weightedMean(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    s = x.shape[-1]
    if weights is None:
        return x.mean(-1)
    return (x * weights).sum(-1) / s
    # non_negative = (samples >= 0).all(-1)
    # samples_log = torch.zeros_like(samples)
    # samples_log[non_negative] = (samples[non_negative] + 1e-12).log()
    # log_mean = (samples_log * weights).sum(-1) / samples.shape[-1]
    # weighted_mean[non_negative] = log_mean[non_negative].exp()


def weightedStd(x: torch.Tensor, weights: torch.Tensor | None = None, n_eff: torch.Tensor | None = None) -> torch.Tensor:
    s = x.shape[-1]
    if weights is None:
        return x.std(-1)
    denom = s
    if n_eff is not None:
        denom = (s - s / n_eff).unsqueeze(-1)
    mean = weightedMean(x, weights)
    d_sq = (x - mean.unsqueeze(-1)).square()
    weighted_d_sq = d_sq * weights
    weighted_var = weighted_d_sq.sum(-1) / (denom + 1e-6)
    return weighted_var.sqrt()


