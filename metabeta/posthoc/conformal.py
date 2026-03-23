import numpy as np
import torch
from pathlib import Path

from metabeta.evaluation.intervals import ALPHAS, getCredibleIntervals
from metabeta.utils.evaluation import Proposal, getMasks

CIDict = dict[str, torch.Tensor]
Corrections = dict[float, CIDict]


