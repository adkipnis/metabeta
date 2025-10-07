import os
from sys import exit
from pathlib import Path
import time
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch import distributions as D

from metabeta.utils import dsFilename, padTensor
from metabeta.data.tasks import MixedEffects
from metabeta.data.markov import fitMFX
from metabeta.data.csv import RealDataset


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate datasets for linear model task."
    )
    parser.add_argument("-t", "--type", type=str, default="mfx", help="Type of dataset (default = mfx)")
    parser.add_argument("--bs_train", type=int, default=4096, help="batch size per training partition (default = 4,096).")
    parser.add_argument("--bs_val", type=int, default=256, help="batch size for validation partition (default = 256).")
    parser.add_argument("--bs_test", type=int, default=128, help="batch size per testing partition (default = 128).")
    parser.add_argument("--bs_load", type=int, default=32, help="Batch size when loading (for grouping n, default = 32)")
    parser.add_argument("--min_m", type=int, default=5, help="MFX: Minimum number of groups (default = 5).")
    parser.add_argument("--max_m", type=int, default=30, help="MFX: Maximum number of groups (default = 30).")
    parser.add_argument("--min_n", type=int, default=10, help="Minimum number of samples per group (default = 10).")
    parser.add_argument("--max_n", type=int, default=70, help="Maximum number of samples per group (default = 70).")
    parser.add_argument("--max_d", type=int, default=3, help="Maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 12).")
    parser.add_argument("--max_q", type=int, default=1, help="Maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).")
    parser.add_argument("--d_tag", type=str, default="gcsemv", help='Suffix for model ID (default = "")')
    parser.add_argument("--toy", action="store_true", help="Generate toy data (default = False)")
    parser.add_argument("--mono", action="store_true", help="Single prior per parameter type (default = False)")
    parser.add_argument("--semi", action="store_false", help="Generate semi-synthetic data (default = False)")
    parser.add_argument("--varied", action="store_true", help="variable d and q (default = False)")
    parser.add_argument("-b", "--begin", type=int, default=0, help="Begin with iteration number #b (default = 0).")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of dataset partitions to generate (default = 100, 0 only generates validation dataset).")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers
def getFileName(ds_type: str, part: int) -> Path:
    if "train" in ds_type:
        size = cfg.bs_train
        b = cfg.bs_load
    elif "val" in ds_type:
        size = cfg.bs_val
        b = 1
    elif "test" in ds_type:
        size = cfg.bs_test
        b = 1
    else:
        raise ValueError

    fn = dsFilename(
        fx_type=cfg.type,
        ds_type=ds_type,
        b=b,
        m=cfg.max_m if cfg.type == "mfx" else 0,
        n=cfg.max_n,
        d=cfg.max_d,
        q=cfg.max_q if cfg.type == "mfx" else 0,
        size=size,
        part=part,
        varied=cfg.varied,
        tag=cfg.d_tag,
        outside=True,
    )
    return fn


def sampleInt(batch_size: int | tuple, min_val: int, max_val: int) -> torch.Tensor:
    """sample batch from discrete uniform
    (include the max_val as possible value)"""
    shape = (batch_size,) if isinstance(batch_size, int) else batch_size
    return torch.randint(min_val, max_val + 1, shape)


def sampleIntBatched(
    min_val: int | torch.Tensor, max_val: int | torch.Tensor
) -> torch.Tensor:
    """batched version of the above with varying bounds"""
    if isinstance(min_val, int):
        min_val = torch.tensor([min_val])
    if isinstance(max_val, int):
        max_val = torch.tensor([max_val])
    n = max(len(min_val), len(max_val))
    samples = min_val + torch.floor((max_val + 1 - min_val) * torch.rand(n)).long()
    return samples


def sampleHN(shape: tuple, scale: float, max_val: float = float("inf")) -> torch.Tensor:
    """sample batch from halfnormal, optionally clamp"""
    val = D.HalfNormal(scale).sample(shape)
    if max_val < float("inf"):
        val = val.clamp(max=max_val) # type: ignore
    return val # type: ignore


# -----------------------------------------------------------------------------
# aggregation

def maxShapes(data: list[dict[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    ''' get the maximal shapes over the whole list for per key '''
    max_shapes = {}
    for entry in data:
        for key, tensor in entry.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("expected all entries to be tensors")
            shape = tuple(tensor.shape)
            if key not in max_shapes:
                max_shapes[key] = shape
            else:
                # Expand max_shapes[key] tuple elementwise to max dimension
                max_shapes[key] = tuple(
                    max(old_dim, new_dim)
                    for old_dim, new_dim in zip(max_shapes[key], shape)
                )
    return max_shapes


def aggregate(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    ''' finds out maximal dimensions for each key in list of tensor-dicts
        collates all tensor-dicts to single tensor-dicts with padded tensors '''
    max_shapes = maxShapes(data)
    batch_size = len(data)
    ds = {
        k: torch.empty(batch_size, *shape, dtype=data[0][k].dtype)
        for k, shape in max_shapes.items()
    }
    for i, item in enumerate(data):
        for k in ds.keys():
            shape = max_shapes[k]
            ds[k][i] = padTensor(item[k], shape)
    return ds


# -----------------------------------------------------------------------------
