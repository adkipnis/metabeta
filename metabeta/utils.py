import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch
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

def dInput(d_data: int, fx_type: str) -> int:
    n_fx = 2 if fx_type == 'mfx' else 1
    return 1 + n_fx * (1 + d_data)

