import argparse
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import time
from collections.abc import Iterable
import torch
from torch import distributions as D
import numpy as np
from scipy.stats import pearsonr
from metabeta.data.dataset import getDataLoader
from metabeta.utils import dsFilename, getConsoleWidth
from metabeta.models.approximators import ApproximatorMFX
from metabeta.evaluation.importance import ImportanceLocal, ImportanceGlobal
from metabeta.evaluation.coverage import getCoverage, plotCalibration, coverageError
from metabeta.evaluation.sbc import getRanks, plotSBC, plotECDF, getWasserstein
from metabeta.evaluation.pp import (
    posteriorPredictiveSample,
    plotPosteriorPredictive,
    weightSubset,
)
from metabeta import plot

CI = [50, 68, 80, 90, 95]
plt.rcParams["figure.dpi"] = 300

###############################################################################


def setup() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--m_tag', type=str, default='r', help='Suffix for model ID (default = '')')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Model seed')
    parser.add_argument('--cores', type=int, default=8, help='Nubmer of processor cores to use (default = 8)')

    # data
    parser.add_argument('--d_tag', type=str, default='r', help='Suffix for data ID (default = '')')
    parser.add_argument('--varied', action='store_true', help='Use data with variable d/q (default = False)')
    parser.add_argument('--semi', action='store_true', help='Use semi-synthetic data (default = True)')
    parser.add_argument('-t', '--fx_type', type=str, default='mfx', help='Type of dataset [ffx, mfx] (default = ffx)')
    parser.add_argument('-d', type=int, default=3, help='Number of fixed effects (with bias, default = 8)')
    parser.add_argument('-q', type=int, default=1, help='Number of random effects (with bias, default = 3)')
    parser.add_argument('-m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('-n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    parser.add_argument('--permute', action='store_false', help='Permute slope variables for uniform learning across heads (default = True)')

    # evaluation
    parser.add_argument('--bs-train', type=int, default=8192, help='macro batch size per training partition (default = 8,192).')
    parser.add_argument('--bs-val', type=int, default=256, help='macro batch size for validation partition (default = 256).')
    parser.add_argument('--bs-test', type=int, default=128, help='macro batch size for test partition (default = 128).')
    parser.add_argument('--bs-mini', type=int, default=32, help='mini batch size (default = 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (Adam, default = 5e-4)')
    parser.add_argument('--standardize', action='store_false', help='Standardize inputs (default = True)')
    parser.add_argument('--importance', action='store_false', help="Do importance sampling (default = True)")
    parser.add_argument('--calibrate', action='store_false', help="Calibrate posterior (default = True)")
    parser.add_argument('--iteration', type=int, default=10, help='Preload model from iteration #p')

    # summary network
    parser.add_argument('--sum_type', type=str, default='set-transformer', help='Summarizer architecture [set-transformer, dual-transformer] (default = set-transformer)')
    parser.add_argument('--sum_blocks', type=int, default=3, help='Number of blocks in summarizer (default = 4)')
    parser.add_argument('--sum_d', type=int, default=128, help='Model dimension (default = 128)')
    parser.add_argument('--sum_ff', type=int, default=128, help='Feedforward dimension (default = 128)')
    parser.add_argument('--sum_depth', type=int, default=1, help='Feedforward layers (default = 1)')
    parser.add_argument('--sum_out', type=int, default=64, help='Summary dimension (default = 64)')
    parser.add_argument('--sum_heads', type=int, default=8, help='Number of heads (poolformer, default = 8)')    
    parser.add_argument('--sum_dropout', type=float, default=0.01, help='Dropout rate (default = 0.01)')
    parser.add_argument('--sum_act', type=str, default='GELU', help='Activation funtction [anything implemented in torch.nn] (default = GELU)')
    parser.add_argument('--sum_sparse', action='store_false', help='Use sparse implementation (poolformer, default = False)')

    # posterior network
    parser.add_argument('--post_type', type=str, default='affine', help='Posterior architecture [affine, spline] (default = affine)')
    parser.add_argument('--flows', type=int, default=3, help='Number of normalizing flow blocks (default = 4)')
    parser.add_argument('--post_ff', type=int, default=128, help='Feedforward dimension (default = 128)')
    parser.add_argument('--post_depth', type=int, default=3, help='Feedforward layers (default = 3)')
    parser.add_argument('--post_dropout', type=float, default=0.01, help='Dropout rate (default = 0.01)')
    parser.add_argument('--post_act', type=str, default='ReLU', help='Activation funtction [anything implemented in torch.nn] (default = ReLU)')

    return parser.parse_args()


# -----------------------------------------------------------------------------
