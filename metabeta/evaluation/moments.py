import torch
from torch.nn import functional as F
from scipy.stats import pearsonr
import numpy as np


Proposed = dict[str, dict[str, torch.Tensor]]

