import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np

from metabeta.simulation import hypersample, Prior, Synthesizer, Emulator, Simulator
from metabeta.utils.io import datasetFilename
from metabeta.utils.sampling import truncLogUni


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hierarchical datasets.')
    # batch dimensions
    parser.add_argument('--bs_train', type=int, default=4096, help='batch size per training partition (default = 4,096).')
    parser.add_argument('--bs_val', type=int, default=256, help='batch size for validation partition (default = 256).')
    parser.add_argument('--bs_test', type=int, default=128, help='batch size per testing partition (default = 128).')
    parser.add_argument('--bs_load', type=int, default=16, help='Batch size when loading (for grouping m, q, d, default = 16)')
    # data dimensions
    parser.add_argument('-d', '--max_d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 16).')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='Maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).')
    parser.add_argument('--min_m', type=int, default=5, help='Minimum number of groups (default = 5).')
    parser.add_argument('--max_m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='Minimum number of samples per group (default = 10).')
    parser.add_argument('--max_n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    # partitions and sources
    parser.add_argument('--partition', type=str, default='val', help='Type of partition in [train, val, test], (default = train)')
    parser.add_argument('-b', '--begin', type=int, default=1, help='Begin generating training epoch number #b.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Total number of training epochs to generate.')
    parser.add_argument('--type', type=str, default='toy', help='Type of predictors [toy, flat, scm, sampled], (default = toy)')
    parser.add_argument('--source', type=str, default='all', help='Source dataset if type==sampled (default = all)')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD if type==sampled (default = False)')
    return parser.parse_args()

# -----------------------------------------------------------------------------
