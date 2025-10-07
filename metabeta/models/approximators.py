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
    "type": "set-transformer",
    "d_model": 64,
    "n_blocks": 3,
    "d_ff": 128,
    "depth": 1,
    "d_output": 64,
    "n_heads": 8,
    "dropout": 0.01,
    "activation": "GELU",
}

posterior_defaults = {
    "type": "flow-affine",
    "flows": 3,
    "d_ff": 128,
    "depth": 3,
    "dropout": 0.01,
    "activation": "ReLU",
}

model_defaults = {"type": "mfx", "seed": 42, "d": 5, "q": 2, "tag": ""}

# -----------------------------------------------------------------------------


# base approximator
