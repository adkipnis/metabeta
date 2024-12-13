from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer

class Base(nn.Module):
    def __init__(self,
                 num_predictors: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int) -> None:
        super(Base, self).__init__()
        self.input_size = num_predictors + 1
        self.hidden_size = hidden_size
        self.output_size = num_predictors
        self.n_layers = n_layers
        self.dropout = dropout
        self.mu = nn.Linear(hidden_size, self.output_size)
        self.logSigma = nn.Linear(hidden_size, self.output_size)
        self.logAB = nn.Linear(hidden_size, 2)
        self.seed = seed

    def initializeWeights(self) -> None:
        ''' Initialize weights using Xavier initialization '''
        torch.manual_seed(self.seed)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor # (batch_size, seq_size, input_size)
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' forward pass, get all intermediate outputs and map them to the parameters of the proposal posterior '''
        outputs = self.internal(x) # (batch_size, seq_size, hidden_size)
        mu = self.mu(outputs) # (batch_size, seq_size, output_size)
        log_sigma = self.logSigma(outputs) # (batch_size, seq_size, output_size)
        log_ab = self.logAB(outputs) # (batch_size, seq_size, 2)
        return mu, log_sigma.exp(), log_ab.exp()


class RNNBase(Base):
    def __init__(self, num_predictors: int, hidden_size: int, n_layers: int, dropout: float, seed: int, last: bool = False) -> None:
        super(RNNBase, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed)
        self.kwargs = {'input_size': self.input_size, 'hidden_size': hidden_size, 'num_layers': n_layers, 'dropout': dropout, 'batch_first': True}

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.model(x)
        return outputs


class GRU(RNNBase):
    def __init__(self, num_predictors: int, hidden_size: int, n_layers: int, dropout: float, seed: int, last: bool = False) -> None:
        super(GRU, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed, last)
        self.model = nn.GRU(**self.kwargs)
        self.initializeWeights()


class LSTM(RNNBase):
    def __init__(self, num_predictors: int, hidden_size: int, n_layers: int, dropout: float, seed: int, last: bool = False) -> None:
        super(LSTM, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed, last)
        self.model = nn.LSTM(**self.kwargs)
        self.initializeWeights()


class TransformerDecoder(Base):
    def __init__(self,
                 num_predictors: int,
                 hidden_size: int,
                 ff_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 n_heads: int = 4,
                 ) -> None:

        super(TransformerDecoder, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed)
        self.embed = nn.Linear(self.input_size, hidden_size)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_size,
                                                dim_feedforward=ff_size,
                                                nhead=n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.model= nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.initializeWeights()

    def causalMask(self, seq_len: int) -> torch.Tensor:
        ''' Create a mask, such that model bases outputs for X[b, i] on X[b, j] for j <= i '''
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) # (batch_size, seq_size, hidden_size)
        causal_mask = self.causalMask(x.size(1)).to(x.device)
        memory = torch.zeros_like(x)
        outputs = self.model(tgt=x, memory=memory, tgt_mask=causal_mask, tgt_is_causal=True)
        return outputs # (batch_size, seq_size, hidden_size)


def main():
    model = TransformerDecoder(3, 256, 512, 1, 0.1, 0)
    mask = model.causalMask(10)
    print(mask)

if __name__ == '__main__':
    main()

