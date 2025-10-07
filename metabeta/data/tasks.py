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

