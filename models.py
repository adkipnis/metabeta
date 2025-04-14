from typing import Union
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer
from torch import distributions as D
from proposal import DiscreteProposal, MixtureProposal, normalBins, halfNormalBins
from utils import dInput

# ------------------------------------------------------------------------
# Building Blocks

class MLP(nn.Module):
    # multi-layer perceptron
    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 d_output: int,
                 act_fn: str):
        super(MLP, self).__init__()
        if act_fn == 'relu':
            act = nn.ReLU()
        elif act_fn == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError("only relu or gelu supported")
            
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            act,
            nn.Linear(d_hidden, d_output)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MAB(nn.Module):
    # Multihead Attention Block
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 act_fn: str = 'gelu',
                 ):
        super(MAB, self).__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.mlp = MLP(d_model, d_ff, d_model, act_fn)
        self.mha = MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z=None):
        if z is None:
            h, _ = self.mha(x, y, y)
        else:
            h, _ = self.mha(x, y, z)
        h = self.ln0(self.lin(x) + h)
        out = self.ln1(h + self.mlp(h))
        return out


# ------------------------------------------------------------------------
# Embedders

class JointEmbedder(nn.Module):
    def __init__(self,
                 d_data: int,
                 d_model: int,
                 fx_type: str):
        super(JointEmbedder, self).__init__()
        self.include_z = fx_type == 'mfx'
        self.d_input = dInput(d_data, fx_type)
        self.emb = nn.Linear(self.d_input, d_model)

    def forward(self, y: torch.Tensor, X: torch.Tensor,
                Z: Union[None, torch.Tensor] = None, **kwargs):
        # assumes y [b, n, 1], x [b, n, d], z [b, n, d]
        if self.include_z:
            inputs = [y, X, Z]
        else:
            inputs = [y, X]
        inputs = torch.cat(inputs, dim=-1)
        out = self.emb(inputs)
        return out


class SeparateEmbedder(nn.Module):
    def __init__(self,
                 d_data: int,
                 d_model: int,
                 fx_type: str):
        super(SeparateEmbedder, self).__init__()
        self.include_z = fx_type == 'mfx'
        if self.include_z:
            d_out = d_model // 3
        else:
            d_out = d_model // 2
        self.emb_y = nn.Linear(1, d_out)
        self.emb_x = nn.Linear(d_data+1, d_out+1)
        if self.include_z:
            self.emb_z = nn.Linear(d_data+1, d_out)

    def forward(self, y: torch.Tensor, X: torch.Tensor,
                Z: Union[None, torch.Tensor] = None, **kwargs):
        y_emb = self.emb_y(y)
        x_emb = self.emb_x(X)
        if self.include_z:
            z_emb = self.emb_z(Z)
            out = [y_emb, x_emb, z_emb]
        else:
            out = [y_emb, x_emb]
        return torch.cat(out, dim=-1)
    

class SequenceEmbedder(nn.Module):
    def __init__(self,
                 d_data: int,
                 d_model: int,
                 max_n: int,
                 fx_type: str,
                 ):
        super(SequenceEmbedder, self).__init__()
        d_input = dInput(d_data, fx_type)
        self.emb_val = nn.Linear(1, d_model)
        self.emb_obs = nn.Embedding(max_n, d_model)
        self.emb_feat = nn.Embedding(d_input, d_model)

    def forward(self, val: torch.Tensor, obs_idx: torch.Tensor,
                feat_idx: torch.Tensor, **kwargs):
        val_emb = self.emb_val(val)
        obs_emb = self.emb_obs(obs_idx)
        feat_emb = self.emb_feat(feat_idx)
        return val_emb + obs_emb + feat_emb


# ------------------------------------------------------------------------
# transformers

class TFE(nn.Module):
    # TransformerEncoder
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int = 4,
                 n_blocks: int = 1,
                 dropout: float = 0.0,
                 act_fn: str = 'gelu',
                 ):
        super(TFE, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                dim_feedforward=d_ff,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation=act_fn)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        if len(x.shape) == 4: # alternatively loop over m
            b, m, n, d_emb = x.shape
            x = x.reshape(-1, n, d_emb)
            padding_mask = padding_mask.reshape(-1, n)
            out = self.net(x, src_key_padding_mask=padding_mask)
            out = out.reshape(b, m, n, d_emb)
        else:
            out = self.net(x, src_key_padding_mask=padding_mask)
        return out


class TFD(nn.Module):
    # TransformerDecoder
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int = 4,
                 n_blocks: int = 1,
                 dropout: float = 0.0,
                 act_fn: str = 'gelu',
                 ):
        super(TFD, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                dim_feedforward=d_ff,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation=act_fn)
        self.net = nn.TransformerDecoder(decoder_layer, num_layers=n_blocks)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        return self.net(x, x, src_key_padding_mask=padding_mask)


# ------------------------------------------------------------------------
# poolers

class PMA(nn.Module):
    # Pooling by Multihead Attention
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 n_seeds: int = 1,
                 dropout: float = 0.0,
                 act_fn: str = 'gelu',
                 ):
        super(PMA, self).__init__()
        self.mab = MAB(d_model, d_ff, n_heads, dropout, act_fn)
        self.seeds = nn.Parameter(torch.Tensor(1, n_seeds, d_model))
        nn.init.xavier_uniform_(self.seeds)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        seeds_tiled = self.seeds.repeat(batch_size, 1, 1)
        return self.mab(seeds_tiled, x)


class PoolingLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, act_fn):
        super(PoolingLayer, self).__init__()
        self.pma = PMA(d_model, d_ff, n_heads, 1, dropout, act_fn=act_fn)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 4: # alternatively loop over m
            b, m, n, d_emb = x.shape
            x = x.reshape(-1, n, d_emb)
            out = self.pma(x).squeeze(-2)
            out = out.reshape(b, m, -1)            
        else:
            out = self.pma(x).squeeze(-2)
        return out


# ------------------------------------------------------------------------
# proposers

class PointPosterior(nn.Module):
    def __init__(self, d_model: int, d_output: int, *args):
        super(PointPosterior, self).__init__()
        self.prop = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor):
        return self.prop(x)

class GeneralizedPosterior(nn.Module):
    # get m linear layers from hidden_size to d_output
    def __init__(self, d_model: int, d_output: int, c: int, m: int = 1):
        super(GeneralizedPosterior, self).__init__()
        self.d_output = d_output
        self.m = m # multiplier
        self.c = c # number of bins / components
        layers = [nn.Linear(d_model, d_output) for _ in range(c * m)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        out = [layer(x).unsqueeze(-1) for layer in self.layers]
        out = torch.cat(out, dim=-1)
        b, d, _ = out.shape
        out = out.reshape(b, d, self.c, -1) # type: ignore
        return out


