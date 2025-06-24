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


# -----------------------------------------------------------------------------
# DeepSet
class InvariantBlock(nn.Module):
    def __init__(self,
                 d_model: int = 64,
                 d_ff: int = 64,
                 dropout: float = 0.05,
                 activation: str = 'Mish',
                 ):
        super().__init__()
        self.phi = MLP(d_model, d_ff, d_model,
                       dropout=dropout, activation=activation, act_on_last=True)
        self.rho = MLP(d_model, d_ff, d_model,
                       dropout=dropout, activation=activation, act_on_last=True)

    def pool(self, h: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is None:
            out = h.mean(dim=-2)
        else:
            out = h * mask.unsqueeze(-1)
            out = out.sum(dim=-2)
            denominator = mask.sum(dim=-1, keepdim=True)
            out = out / (denominator.expand_as(out) + 1e-12)
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x (batch, seq_len, d_model)
        h = self.phi(x)
        h = self.pool(h, mask)
        h = self.rho(h)
        return h


class EquivariantBlock(nn.Module):
    '''
    Steps:
    1. Invariant module (my Deepset) (b, n, emb) -> (b, emb)
    2. tile -> (b, n, emb)
    3. concatenate with input -> (b, n, 2*emb)
    4. project through MLP -> (b, n, emb)
    5. add to initial input -> (b, n, emb)
    6. layer_norm -> (b, n, emb)
    '''
    def __init__(self,
                 d_model: int = 64,
                 d_ff: int = 64,
                 dropout: float = 0.05,
                 activation: str = 'Mish',
                 ):
        super().__init__()
        self.ib = InvariantBlock(d_model, d_ff, dropout, activation)
        self.proj = MLP(2 * d_model, d_ff, d_model,
                        dropout=dropout, activation=activation, act_on_last=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.ib(x, mask)
        h = h.unsqueeze(-2).expand_as(x)
        h_ = torch.cat([x, h], dim=-1)
        h = x + self.proj(h_)
        h = self.norm(h)
        return h


class DeepSet(Summarizer):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 d_output: int,
                 n_blocks: int = 2,
                 dropout: float = 0.05,
                 activation: str = 'Mish',
                 ):
        super().__init__()

        blocks = []
        for _ in range(n_blocks):
            eb = EquivariantBlock(d_model, d_ff, dropout, activation)
            blocks.append(eb)
        self.blocks = nn.ModuleList(blocks)
        self.ib = InvariantBlock(d_model, d_ff, dropout, activation)
        self.out = nn.Linear(d_model, d_output)

    def forward(self, x, mask=None):
        h = x
        for eb in self.blocks:
            h = eb(h, mask)
        h = self.ib(h, mask)
        h = self.out(h)
        return h


