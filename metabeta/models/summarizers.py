from typing import Iterable
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from metabeta.models.feedforward import MLP


# --------------------------------------------------------------------------
# Base Class
class Summarizer(nn.Module):
    ''' takes embedded batch of sequential data x (batch, seq_len, d_model)
    and summarizes the sequences to h (batch, d_output) '''
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


