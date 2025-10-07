import numpy as np
import torch
from torch import nn
from metabeta.utils import (
    maskedMean,
    maskedStd,
    batchCovary,
    maskedSoftplus,
    maskedInverseSoftplus,
    dampen,
    nParams,
)
from metabeta.models.transformers import (
    BaseSetTransformer,
    SetTransformer,
    DualTransformer,
)
from metabeta.models.posteriors import Posterior, CouplingPosterior
from metabeta import plot

mse = nn.MSELoss()

summary_defaults = {
