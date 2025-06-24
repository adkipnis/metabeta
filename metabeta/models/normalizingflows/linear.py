import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class Transform(nn.Module):
    def forward(self,
                x: torch.Tensor,
                condition: torch.Tensor|None = None,
                mask: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError

    def inverse(self,
                z: torch.Tensor,
                condition: torch.Tensor|None = None,
                mask: torch.Tensor|None = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None]:
        raise NotImplementedError

    def forwardMask(self, mask: torch.Tensor):
        return mask


