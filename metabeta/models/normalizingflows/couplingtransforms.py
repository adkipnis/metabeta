from typing import List, Tuple, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.feedforward import MLP, ResidualNet


class CouplingTransform(nn.Module):
    def propose(self,
                x1: torch.Tensor,
                condition: torch.Tensor|None=None
                ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self,
                x2: torch.Tensor,
                params: Dict[str, torch.Tensor],
                mask2: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError

    def inverse(self,
                z2: torch.Tensor,
                params: Dict[str, torch.Tensor],
                mask2: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError


