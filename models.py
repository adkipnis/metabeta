from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer

class Base(nn.Module):
    def __init__(self,
                 num_predictors: int,
                 hidden_size: int,
                 seed: int,
                 reuse: bool = True) -> None:
        super(Base, self).__init__()
        self.input_size = num_predictors + 1
        self.hidden_size = hidden_size
        self.output_size = num_predictors
        self.embedding = nn.Linear(self.input_size, hidden_size)
        self.means = nn.Linear(hidden_size, self.output_size)
        # self.logstds = nn.Linear(hidden_size, output_size * (output_size + 1) // 2)
        self.logstds = nn.Linear(hidden_size, self.output_size)
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

    def toCovariance(self, logstds: torch.Tensor) -> torch.Tensor:
        ''' construct triangular matrix from logstds, apply softplus to diagonal and multiply with its transpose '''
        # logstds (b, output_size_sigma) or (b, n, output_size_sigma)
        d = self.output_size # number of predictors
        b = logstds.size(0) # batch size
        tril_ids = torch.tril_indices(d, d, offset=0)
        diag_ids = torch.arange(d)
        if logstds.dim() == 2:
            chol_matrix = torch.zeros(b, d, d, device=logstds.device)
            chol_matrix[:, tril_ids[0], tril_ids[1]] = logstds
            chol_matrix[:, diag_ids, diag_ids] = F.softplus(chol_matrix[:, diag_ids, diag_ids])
        else:
            n = logstds.size(1) # number of samples
            chol_matrix = torch.zeros(b, n, d, d, device=logstds.device)
            chol_matrix[:, :, tril_ids[0], tril_ids[1]] = logstds
            chol_matrix[:, :, diag_ids, diag_ids] = F.softplus(chol_matrix[:, :, diag_ids, diag_ids])
        cov_matrix = chol_matrix @ chol_matrix.transpose(-1, -2)
        if logstds.dim() == 2:
            cov_matrix[:, diag_ids, diag_ids] += 1e-6 # add small value to diagonal
        else:
            cov_matrix[:, :, diag_ids, diag_ids] += 1e-6
        return cov_matrix
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' forward pass, get all intermediate outputs and map them to the parameters of the proposal posterior '''
        # x (batch_size, seq_size, input_size)
        x = self.embedding(x) # (batch_size, seq_size, hidden_size)
        outputs = self.internal(x) # (batch_size, seq_size, hidden_size)

        # Forward pass through TransformerDecoder
        if not self.reuse:
            outputs = outputs[:, -1] # (batch_size, hidden_size)
        
        # Transform outputs
        means = self.means(self.relu(outputs)) # (batch_size, output_size) or (batch_size, seq_size, output_size)
        logstds = self.logstds(self.relu(outputs))
        # sigmas = self.toCovariance(logstds) # (batch_size, output_size, output_size) or (batch_size, seq_size, output_size, output_size)
        return means, logstds.exp()


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


