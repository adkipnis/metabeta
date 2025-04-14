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


    def __init__(self,
                 n_inputs: int,
                 n_predictors: int, # including bias term
                 hidden_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 fx_type: str,
                 posterior_type: str,
                 n_components: int = 1) -> None:
        super(Base, self).__init__()
        self.n_inputs = n_inputs
        self.n_predictors = n_predictors
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.seed = seed
        self.fx_type = fx_type
        self.posterior_type = posterior_type
        self.n_components = n_components # number of mixture components resp. bins

        # embeddings
        self.embed_y = nn.Linear(1, hidden_size)
        self.embed_x = nn.Linear(n_predictors, hidden_size)
        self.embed_z = nn.Linear(n_predictors, hidden_size)
        self.embed_g = nn.Embedding(5, hidden_size) # todo: more groups

        # posterior
        n_fx = 1 if fx_type == "ffx" else 2
        if posterior_type == "discrete":
            self.final_layers = generalizedPosterior(hidden_size, n_predictors, n_fx * self.n_components) # (ffx, rfx) * (bin)
        elif posterior_type == "mixture":
            self.final_layers = generalizedPosterior(hidden_size, n_predictors, 3 * n_fx * self.n_components) # (loc, scale, weight) * (ffx, rfx) * (component)
        elif posterior_type == "discrete_noise":
            self.final_layers = generalizedPosterior(hidden_size, 1, self.n_components)
        elif posterior_type == "mixture_noise":
            self.final_layers = generalizedPosterior(hidden_size, 1, 3 * self.n_components)
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

