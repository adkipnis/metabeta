import torch
from torch.nn import functional as F
from scipy.stats import pearsonr
import numpy as np


Proposed = dict[str, dict[str, torch.Tensor]]

def numFixed(proposed: Proposed) -> int:
    q = proposed['local']['samples'].shape[-1]
    D = proposed['global']['samples'].shape[-1]
    d = D - q - 1
    return int(d)

