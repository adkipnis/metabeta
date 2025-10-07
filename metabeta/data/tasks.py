import torch
from torch import distributions as D
import random
from metabeta.data.distributions import (
    Normal,
    StudentT,
    Uniform,
    Bernoulli,
    NegativeBinomial,
    ScaledBeta,
)
from metabeta.utils import fullCovary

# -----------------------------------------------------------------------------
probs = torch.tensor([0.10, 0.40, 0.05, 0.25, 0.10, 0.10])
dists = [
    Normal,
    StudentT,
    Uniform,
    Bernoulli,
    NegativeBinomial,
    ScaledBeta,
]


