import os
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
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
def maskedMean(x: torch.Tensor, dim: tuple | int, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        sums = x.sum(dim, keepdim=True)
        count = mask.sum(dim, keepdim=True)
        return sums / (count + 1e-12)
    else:
        return x.mean(dim, keepdim=True)


def maskedStd(
    x: torch.Tensor, dim: tuple, mask: torch.Tensor | None, mean: torch.Tensor | None = None
) -> torch.Tensor:
    if mean is None:
        mean = maskedMean(x, dim, mask)
    if mask is not None:
        diff_squared_sum = ((x - mean) * mask).square().sum(dim, keepdim=True)
        count = mask.sum(dim, keepdim=True) - 1
        count = torch.where(count < 1, 1, count)
        return (diff_squared_sum / count).sqrt()
    else:
        return x.std(dim, keepdim=True)


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


# -----------------------------------------------------------------------------
# regularization utils
shift = 1e-3


def maskedLog(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, x.log(), 0)


def inverseSoftPlus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20, x, torch.log(torch.exp(x) - 1))


def maskedSoftplus(x: torch.Tensor, eps: float = shift) -> torch.Tensor:
    return torch.where(x != 0, F.softplus(x + eps), 0)


def maskedInverseSoftplus(x: torch.Tensor, eps: float = shift) -> torch.Tensor:
    return torch.where(x != 0, inverseSoftPlus(x) - eps, 0)


def dampen(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    return x.sign() * x.abs().pow(p)


def squish(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


# -----------------------------------------------------------------------------
# permutation


def getPermutation(d: int):
    p = torch.randperm(d - 1) + 1
    zero = torch.zeros((1,), dtype=p.dtype)
    p = torch.cat([zero, p])
    return p

def inversePermutation(p: torch.Tensor):
    q = torch.empty_like(p)
    q[p] = torch.arange(p.size(0), dtype=p.dtype)
    return q


# -----------------------------------------------------------------------------
# batch handling


def _copy(this):
    if isinstance(this, torch.Tensor):
        out = this.clone()
    elif isinstance(this, dict):
        out = this.copy()
    else:
        out = this
    return out


def catDict(big: dict, small: dict) -> dict:
    for k in small:
        entry = _copy(small[k])
        if k not in big:
            big[k] = entry
        elif isinstance(entry, torch.Tensor):
            assert isinstance(big[k], torch.Tensor), (
                "expected target point to be a list"
            )
            big[k] = torch.cat([big[k], entry])
        elif isinstance(entry, dict):
            assert isinstance(big[k], dict), "expected target point to be a dict"
            big[k] = catDict(big[k], entry)
        elif isinstance(entry, np.ndarray):
            assert all(big[k] == entry), "np arrays differ"
        elif entry is None:
            assert big[k] is None, f"{k} is not None but new entry is"
        else:
            raise ValueError("entry is neither a tensor nor a dict")
    return big


def padTensor(tensor: torch.Tensor, shape: tuple, value=0) -> torch.Tensor:
    assert len(tensor.shape) == len(shape), (
        "Input tensor and target shape must have the same number of dimensions"
    )
    shapes = zip(reversed(shape), reversed(tensor.shape))
    pad_size = [max(s - t, 0) for s, t in shapes]
    padding = []
    for p in pad_size:
        padding.extend([0, p])  # Pad at the end of each dimension
    out = F.pad(tensor, padding, value=value)
    return out


def fullCovary(data: torch.Tensor) -> torch.Tensor:
    # data (n, q)
    n = data.shape[0]
    mean = data.mean(0, keepdim=True)
    centered = data - mean
    cov = (centered.transpose(0, 1) @ centered) / (n - 1)
    return cov


def batchCovary(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # data (b, m, q), mask (b, m)
    m = mask.sum(1)
    denom = (m - 1).clamp(min=1).view(-1, 1, 1)
    mean = maskedMean(data, 1, mask.unsqueeze(-1))  # data.mean(1, keepdim=True)
    centered = data - mean
    cov = (centered.transpose(1, 2) @ centered) / denom
    return cov


# -----------------------------------------------------------------------------
# plotting utils
cmap = plt.get_cmap("tab20")
palette = [mcolors.to_hex(cmap(i)) for i in range(0, cmap.N, 2)]
palette += [mcolors.to_hex(cmap(i)) for i in range(1, cmap.N, 2)]

# -----------------------------------------------------------------------------
# profiling
def check(x: torch.Tensor) -> None:
    if x.isnan().any():
        idx = torch.where(x.isnan())
        print("nans at", idx)
    if x.abs().isinf().any():
        idx = torch.where(x.abs().isinf())
        print("infs at", idx)


def profile(fn, inputs):
    import torch.profiler

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        fn(*inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def printRAM():
    import gc

    print("\nLargest tensors by size (bytes):")
    tensors = [
        (obj.element_size() * obj.nelement(), obj.shape)
        for obj in gc.get_objects()
        if torch.is_tensor(obj)
    ]
    tensors.sort(key=lambda x: x)
    for size, shape in tensors[-1:-20:-1]:
        print(f"Shape: {shape}, Size: {size / 1_048_576:.2f} mb")


