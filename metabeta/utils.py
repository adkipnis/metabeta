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


