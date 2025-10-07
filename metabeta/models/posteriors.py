import numpy as np
import torch
from torch import nn
from metabeta.models.normalizingflows.coupling import CouplingFlow
from metabeta.utils import weightedMean, weightedStd
from metabeta.evaluation.coverage import Calibrator
from metabeta import plot

mse = nn.MSELoss(reduction="none")


