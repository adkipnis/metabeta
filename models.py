from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoderLayer

class Base(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 seed: int,
                 reuse: bool = True) -> None:
        super(Base, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.means = nn.Linear(hidden_size, output_size)
        # self.logstds = nn.Linear(hidden_size, output_size * (output_size + 1) // 2)
        self.logstds = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.seed = seed
        self.reuse = reuse # reuse intermediate outputs
        self.initializeWeights()

    def initializeWeights(self) -> None:
        ''' Initialize weights using Xavier initialization '''
        seed = self.seed
        for p in self.parameters():
            if p.dim() > 1:
                torch.manual_seed(seed)
                seed += 1
                nn.init.xavier_uniform_(p)

    def createSigma(self, rootstds: torch.Tensor) -> torch.Tensor:
        # rootstds (batch_size, output_size_sigma)
        # construct triangular matrix from rootstds and multiply by its transpose
        triangular = torch.zeros(rootstds.size(0), self.output_size, self.output_size, device=rootstds.device)
        idx = torch.tril_indices(self.output_size, self.output_size)
        triangular[:, idx[0], idx[1]] = rootstds
        return torch.bmm(triangular, triangular.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x (batch_size, seq_size, input_size)
        x = self.embedding(x) # (batch_size, seq_size, hidden_size)
        outputs = self.internal(x) # (batch_size, seq_size, hidden_size)

        # Forward pass through TransformerDecoder
        if not self.reuse:
            outputs = outputs[:, -1] # (batch_size, hidden_size)
        
        # Transform outputs
        means = self.means(self.relu(outputs)) # (batch_size, output_size) or (batch_size, seq_size, output_size)
        rootstds = self.logstds(self.relu(outputs))
        sigmas = self.createSigma(rootstds) # (batch_size, output_size, output_size) or (batch_size, seq_size, output_size, output_size)
        return means, sigmas


class RNN(Base):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(RNN, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.model = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, seq_size, hidden_size)
        outputs, _ = self.model(x)
        return outputs


class GRU(RNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(GRU, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.model = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)


class LSTM(RNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(LSTM, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.model = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)


class TransformerDecoder(Base):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 ff_size: int,
                 output_size: int,
                 seed: int,
                 reuse: bool = True,
                 nhead: int = 4,
                 num_layers: int = 1,
                 ) -> None:

        super(TransformerDecoder, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=ff_size, batch_first=True)
        self.model= nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, seq_size, hidden_size)
        return self.model(x, x)


