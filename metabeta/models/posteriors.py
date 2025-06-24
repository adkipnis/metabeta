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

# -----------------------------------------------------------------------------
# helpers
def inverseSoftplus(y: torch.Tensor) -> torch.Tensor:
    return torch.where(
        y == 0.,
        torch.zeros_like(y),
        torch.where(
            y > 20.,
            y,
            torch.log(torch.expm1(y))
        )
    )
        

class GeneralizedProposer(nn.Module):
    # get m linear layers from hidden_size to d_output
    def __init__(self, d_model: int, d_output: int, c: int, m: int = 1):
        super().__init__()
        self.d_output = d_output
        self.m = m # multiplier
        self.c = c # number of bins / components
        layers = [nn.Linear(d_model, d_output) for _ in range(c * m)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        out = [layer(x).unsqueeze(-1) for layer in self.layers]
        out = torch.cat(out, dim=-1)
        if out.dim() == 3:
            b, d = out.shape[:2]
            out = out.reshape(b, d, self.c, -1) # type: ignore
        else:
            b, m, d = out.shape[:3]
            out = out.reshape(b, m, d, self.c, -1) # type: ignore
        return out



