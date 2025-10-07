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
