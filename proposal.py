from math import sqrt
import torch
from torch import nn
from torch import distributions as D
mse = nn.MSELoss(reduction='none')

# --------------------------------------------------------------------------
# discrete posterior
def equidistantBins(low: float, high: float, n_bins: int) -> torch.Tensor:
    return torch.linspace(low, high, steps=n_bins)

def normalBins(scale: float, n_bins: int) -> torch.Tensor:
    quantiles = torch.linspace(0., 1., n_bins+2)
    quantiles = quantiles[1:-1]
    return D.Normal(0, scale).icdf(quantiles)

def halfNormalBins(scale: float, n_bins: int) -> torch.Tensor:
    quantiles = torch.linspace(0., 1., n_bins+1)
    quantiles = quantiles[:-1]
    return D.HalfNormal(scale).icdf(quantiles)

