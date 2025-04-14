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


        else:
            raise ValueError(f"model type: {posterior_type} not supported")

    def initializeWeights(self) -> None:
        ''' Initialize weights using Xavier initialization '''
        torch.manual_seed(self.seed)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, y: torch.Tensor, X: torch.Tensor,
              Z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        y_emb = self.embed_y(y)
        X_emb = self.embed_x(X)
        Z_emb = self.embed_z(Z)
        g_emb = self.embed_g(groups)
        return y_emb + X_emb + Z_emb + g_emb

    def forward(self, y: torch.Tensor, X: torch.Tensor,
                Z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        ''' forward pass, get all intermediate outputs and map them to the parameters of the proposal posterior '''
        outputs = self.internal(y, X, Z, groups)
        finals = [layer(outputs).unsqueeze(-1) for layer in self.final_layers]
        return torch.cat(finals, dim=-1)


class TransformerEncoder(Base):
    def __init__(self,
                 n_inputs: int,
                 n_predictors: int,
                 hidden_size: int,
                 ff_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 n_heads: int = 4,
                 fx_type: str = "ffx",
                 posterior_type: str = "mixture",
                 n_components: int = 1,
                 activation: str = 'gelu',
                 ) -> None:

        super(TransformerEncoder, self).__init__(n_inputs, n_predictors, hidden_size, n_layers, dropout, seed, fx_type, posterior_type, n_components)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size,
                                                dim_feedforward=ff_size,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation=activation)
        self.model= nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.initializeWeights()

    def internal(self, y: torch.Tensor, X: torch.Tensor,
                 Z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        o = self.embed(y, X, Z, groups) # (batch_size, seq_size, hidden_size)
        mask = causalMask(o.size(1)).to(o.device)
        outputs = self.model(o, mask)
        return outputs # (batch_size, seq_size, hidden_size)


class TransformerDecoder(Base):
    def __init__(self,
                 n_inputs: int,
                 n_predictors: int,
                 hidden_size: int,
                 ff_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 n_heads: int = 4,
                 fx_type: str = "ffx",
                 posterior_type: str = "mixture",
                 n_components: int = 1,
                 activation: str ='gelu',
                 ) -> None:

        super(TransformerDecoder, self).__init__(n_inputs, n_predictors, hidden_size, n_layers, dropout, seed, fx_type, posterior_type, n_components)
        self.embed = nn.Linear(n_inputs, hidden_size)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_size,
                                                dim_feedforward=ff_size,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation=activation)
        self.model= nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.initializeWeights()

    def internal(self, y: torch.Tensor, X: torch.Tensor,
                 Z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        o = self.embed(y, X, Z, groups) # (batch_size, seq_size, hidden_size)
        causal_mask = causalMask(o.size(1)).to(o.device)
        memory = torch.zeros_like(o)
        outputs = self.model(tgt=o, memory=memory, tgt_mask=causal_mask, tgt_is_causal=True)
        return outputs # (batch_size, seq_size, hidden_size)


def main():
    emod = TransformerEncoder(6, 3, 256, 512, 1, 0.1, 0)
    dmod = TransformerEncoder(6, 3, 256, 512, 1, 0.1, 0)
    mask = causalMask(10)
    print(mask)

if __name__ == '__main__':
    main()

