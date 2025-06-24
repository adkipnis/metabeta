import argparse
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde, binom
import torch
from torch import distributions as D
from metabeta.utils import dsFilename, getConsoleWidth, modelID, getDataLoader
from metabeta.models.approximators import Approximator, ApproximatorFFX#, ApproximatorMFX
from metabeta.importance import importanceSample
plt.rcParams['figure.dpi'] = 300
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(cmap.N)]

# --- for given snapshot:
# 1. evaluate performance (over batch):
# - plot recovery  [✓]
# - plot error bars [✓]
# - add rmse + r  [✓]

# 2. calibration
# - coverage over different CI's [✓]
# - plot calibration curve (nominal vs. empirical) [✓]
# - SBC histogram [✓] 
# - SBC shade 
# - ECDF difference plot [✓] 

# 3. visualize posterior (conditional on single dataset):
# - plot posterior distribution with vline for true values [✓]
# - (1) mixture [✓]
# - (2) discrete [✓]
# - (3) flow [✓]

# 4. visualize posterior devolopment over n (conditional on single dataset):
# - evaluate quantiles [✓]
# - subset dataset and run model on each [✓]
# - plot quantiles [✓]

# 5. posterior predictive checks
# - read up on theory

# --- over snapshots:
# 1. visualize training and testing NLLL [✓]
# 2. evaluate [✓] and visualize KLD [✓]

# -----------------------------------------------------------------------------
# coverage

def empiricalCoverage(quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #  how often are the targets actually inside a given credibility interval?
    mask = (targets != 0.)
    above = targets > quantiles[..., 0]
    below = targets < quantiles[..., -1]
    inside = above * below * mask
    coverage = inside.float().sum(0)/(mask.sum(0) + 1e-12)
    return coverage # (d,)

def getCoverage(model: Approximator,
                proposed: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    quantiles50 = model.quantiles(proposed, [.25, .75])
    quantiles68 = model.quantiles(proposed, [.16, .84])
    quantiles80 = model.quantiles(proposed, [.10, .90])
    quantiles90 = model.quantiles(proposed, [.05, .95])
    quantiles95 = model.quantiles(proposed, [.025, .975])
    c50 = empiricalCoverage(quantiles50, targets)
    c68 = empiricalCoverage(quantiles68, targets)
    c80 = empiricalCoverage(quantiles80, targets)
    c90 = empiricalCoverage(quantiles90, targets)
    c95 = empiricalCoverage(quantiles95, targets)
    return {'50': c50, '68': c68, '80': c80, '90': c90, '95': c95}

