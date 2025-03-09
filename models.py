from typing import List
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

def getLinearLayers(d_in: int, d_out: int, n_rep: int) -> nn.ModuleList:
    layers = [nn.Linear(d_in, d_out) for _ in range(n_rep)]
    return nn.ModuleList(layers)

def parametricPosterior(hidden_size: int, d: int) -> nn.ModuleList:
    ffx_layers = getLinearLayers(hidden_size, d, 2)
    rfx_layers = getLinearLayers(hidden_size, d, 2)
    return nn.ModuleList(list(ffx_layers) + list(rfx_layers))

def generalizedPosterior(hidden_size: int, d: int, m: int) -> nn.ModuleList:
    return getLinearLayers(hidden_size, d, m)

# def causalMask(seq_len: int) -> torch.Tensor:
#     ''' Create a mask, such that model bases outputs for X[b, i] on X[b, j] for j <= i '''
#     return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

def causalMask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))



class Base(nn.Module):
    def __init__(self,
                 n_predictors: int, # including bias term
                 hidden_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 fx_type: str,
                 posterior_type: str,
                 n_components: int = 1) -> None:
        super(Base, self).__init__()
        self.n_predictors = n_predictors
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.seed = seed
        self.fx_type = fx_type
        self.posterior_type = posterior_type
        self.n_components = n_components # number of mixture components
        n_fx = 1 if fx_type == "ffx" else 2
        if posterior_type == "parametric":
            self.final_layers = parametricPosterior(hidden_size, n_predictors)
        elif posterior_type == "discrete":
            self.final_layers = generalizedPosterior(hidden_size, n_predictors, n_fx * self.n_components) # (ffx, rfx) * (bin)
        elif posterior_type == "mixture":
            self.final_layers = generalizedPosterior(hidden_size, n_predictors, 3 * n_fx * self.n_components) # (loc, scale, weight) * (ffx, rfx) * (mixture component)
        elif posterior_type == "parametric_noise":
            self.final_layers = getLinearLayers(hidden_size, 2, 1)
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

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self,
                x: torch.Tensor, # (batch_size, seq_size, input_size)
                ) -> torch.Tensor:
        ''' forward pass, get all intermediate outputs and map them to the parameters of the proposal posterior '''
        outputs = self.internal(x) # (batch_size, seq_size, hidden_size)
        finals = [layer(outputs).unsqueeze(-1) for layer in self.final_layers]
        return torch.cat(finals, dim=-1)


class TransformerDecoder(Base):
    def __init__(self,
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
                 ) -> None:

        super(TransformerDecoder, self).__init__(n_predictors, hidden_size, n_layers, dropout, seed, fx_type, posterior_type, n_components)
        self.embed = nn.Linear(n_predictors+1, hidden_size)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_size,
                                                dim_feedforward=ff_size,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.model= nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.initializeWeights()

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) # (batch_size, seq_size, hidden_size)
        causal_mask = causalMask(x.size(1)).to(x.device)
        memory = torch.zeros_like(x)
        outputs = self.model(tgt=x, memory=memory, tgt_mask=causal_mask, tgt_is_causal=True)
        return outputs # (batch_size, seq_size, hidden_size)


def main():
    model = TransformerDecoder(3, 256, 512, 1, 0.1, 0)
    mask = model.causalMask(10)
    print(mask)

if __name__ == '__main__':
    main()

