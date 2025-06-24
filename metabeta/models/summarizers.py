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


# -----------------------------------------------------------------------------
# PoolFormer
class MAB(nn.Module):
    # Multihead Attention Block
    def __init__(self,
                 d_model: int,
                 d_hidden: int | Iterable,
                 n_heads: int = 4,
                 activation: str = 'GELU',
                 dropout: float = 0.01,
                 ):
        super(MAB, self).__init__()
        
        # projection
        self.mlp = MLP(
            d_input=d_model, d_hidden=d_hidden, d_output=d_model,
            activation=activation, dropout=dropout)
        
        # attention
        self.mha = MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, add_bias_kv=True)
        
        # layer norms
        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor | None = None):
        h, _ = self.mha(x, y, y, key_padding_mask=mask)
        h = self.ln0(h + x)
        out = self.ln1(h + self.mlp(h))
        return out


class PMA(nn.Module):
    # Pooling by Multihead Attention
    def __init__(self, 
                 d_model: int,
                 d_hidden: int | Iterable,
                 n_heads: int = 4,
                 dropout: float = 0.01,
                 activation: str = 'GELU',
                 n_seeds: int = 1,
                 ):
        super(PMA, self).__init__()
        
        self.enc = MLP(
            d_input=d_model, d_hidden=d_hidden, d_output=d_model,
            activation=activation, dropout=dropout)
        self.mab = MAB(
            d_model=d_model, d_hidden=d_hidden, n_heads=n_heads,
            activation=activation, dropout=dropout)
        self.s = nn.Parameter(torch.Tensor(1, n_seeds, d_model))
        nn.init.xavier_uniform_(self.s)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.enc(x)
        seeds_tiled = self.s.repeat(x.size(0), 1, 1)
        out = self.mab(seeds_tiled, h, mask=mask)
        return out


class TFE(nn.Module):
    # TransformerEncoder
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int = 4,
                 n_blocks: int = 1,
                 dropout: float = 0.01,
                 activation: str = 'GELU',
                 ):
        super(TFE, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                dim_feedforward=d_ff,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation=activation.lower())
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        return self.net(x, src_key_padding_mask=padding_mask)


class PoolFormer(Summarizer):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 d_output: int,
                 depth: int = 2,
                 n_heads: int = 4,
                 n_blocks: int = 2,
                 dropout: float = 0.01,
                 activation: str = 'GELU',
                 ):
        super().__init__()
        self.d_output = d_output
        self.enc = TFE(d_model, d_ff, n_heads, n_blocks,
                       dropout=dropout, activation=activation)
        self.pma = PMA(d_model, (d_ff,)*depth, n_heads,
                       dropout=dropout, activation=activation)
        self.out = nn.Linear(d_model, d_output)
        
    def prepareMask(self, mask: torch.Tensor):
        mask = ~mask.bool()
        if mask.dim() > 2:
            b, m, n = mask.shape
            mask = mask.view(b*m, n)
        non_empty = ~mask.all(dim=1)
        idx = non_empty.nonzero(as_tuple=True)[0]
        mask = mask[idx]
        return mask, idx
    
    def prepareX(self, x: torch.Tensor, idx: torch.Tensor | None = None):
        if x.dim() > 3:
            b, m, n, d = x.shape
            x = x.view(b*m, n, d)
        if idx is not None:
            x = x[idx]
        return x
    
    def postProcess(self, h, x, idx: torch.Tensor | None = None):
        if x.dim() == 3:
            return h
        b, m, _, _ = x.shape
        if idx is not None:
            h_ = torch.zeros((b*m, self.d_output), device=x.device)
            h_[idx] = h
        else:
            h_ = h
        out = h_.view(b, m, -1)
        return out
    
    def forward(self, x, mask=None):
        mask_, idx = None, None
        if mask is not None:
            mask_, idx = self.prepareMask(mask)
        x_ = self.prepareX(x, idx)
        h = self.enc(x_, mask_)
        h = self.pma(h, mask_).squeeze(-2)
        h = self.out(h)
        out = self.postProcess(h, x, idx)
        return out


