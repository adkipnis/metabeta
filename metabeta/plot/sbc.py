from pathlib import Path
import torch
import numpy as np
from scipy.stats import binom
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


from metabeta.utils.evaluation import getAllNames, getMasks, Proposal, joinSigmas


def fractionalRanks(
    samples: torch.Tensor,  # (b, ..., s, d)
    targets: torch.Tensor,  # (b, ..., d)
) -> torch.Tensor:
    sample_dim = -1 if targets.dim() == 1 else -2
    targets = targets.unsqueeze(sample_dim)
    smaller = samples < targets
    return smaller.float().mean(sample_dim)


def getFractionalRanks(
    proposal: Proposal, data: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    param_names = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
    ranks = {}
    for param in param_names:
        samples = getattr(proposal, param)
        targets = data[param]
        ranks[param] = fractionalRanks(samples, targets)
    return ranks


def theoreticalLimits(
    b: int,  # number of posteriors (batch size)
    s: int = 1000,  # number of samples per posterior
    alpha: float = 0.05,  # alpha error for bounds
    diff: bool = False,
) -> tuple:
    p = np.linspace(0, 1, b)
    lower = binom.ppf(alpha / 2, s, p) / s
    upper = binom.ppf(1 - alpha / 2, s, p) / s
    if diff:
        lower -= p
        upper -= p
    return p, lower, upper


