import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde, binom


def getRanks(
    targets: torch.Tensor, proposed: dict, absolute=False, mask_0=False
) -> torch.Tensor:
    # targets (b, d), samples (b, s, d)
    mask = targets != 0
    if mask_0:  # ignore d=1 models
        mask[:, 0] = mask[:, 1:-1].any(1)
    samples = proposed["samples"]
    weights = proposed.get("weights", None)
    targets_ = targets.unsqueeze(-1)
    if absolute:
        samples = samples.abs()
        targets_ = targets_.abs()
    smaller = (samples <= targets_).float()
    if weights is not None:
        ranks = (smaller * weights).sum(-1) / samples.shape[-1]
    else:
        ranks = smaller.mean(-1)
    ranks[~mask] = -1
    return ranks


def plotSBC(
