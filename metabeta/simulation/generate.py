from sys import exit
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hierarchical datasets.')
    # batch dimensions
    parser.add_argument('--bs_train', type=int, default=4096, help='batch size per training partition (default = 4,096).')
    parser.add_argument('--bs_val', type=int, default=256, help='batch size for validation partition (default = 256).')
    parser.add_argument('--bs_test', type=int, default=128, help='batch size per testing partition (default = 128).')
    parser.add_argument('--bs_load', type=int, default=16, help='Batch size when loading (for grouping n, default = 16)')
    # data dimensions
    parser.add_argument('-d', '--max_d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes) to draw per linear model.')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='Maximum number of random effects (intercept + slopes) to draw per linear model.')
    parser.add_argument('--min_m', type=int, default=5, help='Minimum number of groups (default = 5).')
    parser.add_argument('--max_m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='Minimum number of samples per group (default = 10).')
    parser.add_argument('--max_n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    # partitions and sources
    parser.add_argument('--partition', type=str, default='train', help='Type of partition in [train, val, test], (default = train)')
    parser.add_argument('-b', '--begin', type=int, default=1, help='Begin with training epoch number #b.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to generate.')
    parser.add_argument('--type', type=str, default='semi', help='Type of data [semi, toy, subset], (default = semi)')
    parser.add_argument('--source', type=str, default='all', help='Source tag for non-toy data (default = all)')
    return parser.parse_args()

