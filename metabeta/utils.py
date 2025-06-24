import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from torch import nn
from metabeta.dataset import LMDataset


# ----------------------------------------------------------------------------
# name utils
def getConsoleWidth() -> int:
    ''' Determine the width of the console for formatting purposes. '''
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        print("Could not determine console width. Defaulting to 80.")
        console_width = 80
    return console_width

def parseSize(size: int) -> str:
    if size >= 1e6:
        n = f'{size/1e6:.0f}m'
    elif size >= 1e3:
        n = f'{size/1e3:.0f}k'
    else:
        n = str(size)
    return n

# -----------------------------------------------------------------------------
# loading utils

def setDevice(device: str):
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def dsFilename(fx_type: str, ds_type: str, d: int, m: int, n: int, size: int, part: int, ftype: str = 'pt') -> Path:
    ''' example: ffx-train-d=8-m=1-n=50-50k-001.pt'''
    s = parseSize(size)
    p = f'{part:03d}'
    return Path('data', f'{fx_type}-{ds_type}-d={d}-m={m}-n={n}-{s}-{p}.{ftype}')

def modelID(cfg: argparse.Namespace) -> str:
    ''' Return a string that identifies the model. '''
    c = 1
    if cfg.post_type == "discrete":
        c = cfg.bins
    elif cfg.post_type == "mixture":
        c = cfg.components
    elif cfg.post_type in ["flow-affine", "flow-spline", "flow-matching"]:
        c = cfg.flows
    return f"{cfg.fx_type}-{cfg.emb_type}-{cfg.sum_type}-{cfg.blocks}-{cfg.hidden}-{cfg.ff}-{cfg.out}-{cfg.post_type}-{c}-{cfg.act}-dropout={cfg.dropout}-seed={cfg.seed}"

def getDataSet(filename: Path, permute: bool = True) -> LMDataset:
    ''' Load a dataset from a file, optionally flatten data to sequence, optionally permute features, return dataloader'''
    assert filename.exists(), f"File {filename} does not exist, you must generate it first using generate.py"
    ds_raw = torch.load(filename, weights_only=False, mmap=True)
    return LMDataset(**ds_raw, permute=permute)

def getDataLoader(filename: Path, batch_size: int, permute: bool = True) -> DataLoader:
    ds = getDataSet(filename, permute)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# -----------------------------------------------------------------------------
# size utils
def dInput(d_data: int, fx_type: str) -> int:
    n_fx = 2 if fx_type == 'mfx' else 1
    return 1 + n_fx * (1 + d_data)

def nParams(model: nn.Module) -> None:
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

