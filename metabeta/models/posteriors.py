from typing import Tuple, Type, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import distributions as D
import torch.nn.functional as F
from metabeta.models.normalizingflows.coupling import CouplingFlow
from metabeta.models.normalizingflows.flowmatching import FlowMatching
from metabeta.utils import maskLoss
plt.rcParams.update({'font.size': 16})
mse = nn.MSELoss(reduction='none')

