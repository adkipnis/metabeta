from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Base(nn.Module):
    def __init__(self,
                 num_predictors: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 last: bool = False) -> None:
        super(Base, self).__init__()
        self.input_size = num_predictors + 1
        self.hidden_size = hidden_size
        self.output_size = num_predictors
        self.n_layers = n_layers
        self.dropout = dropout
        self.means = nn.Linear(hidden_size, self.output_size)
        # self.logstds = nn.Linear(hidden_size, output_size * (output_size + 1) // 2)
        self.logstds = nn.Linear(hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.seed = seed
        self.last = last
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

    def internal(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' forward pass, get all intermediate outputs and map them to the parameters of the proposal posterior '''
        # x (batch_size, seq_size, input_size)
        
        # run through main layers
        outputs = self.internal(x, lengths) # (batch_size, seq_size, hidden_size) if no masking
    
        # optionally take last output
        if self.last:
            batch_size = lengths.size(0)
            outputs = outputs[torch.arange(batch_size), lengths-1] # (batch_size, hidden_size)
        
        # Transform outputs
        means = self.means(self.relu(outputs)) # (batch_size, output_size) or (batch_size, seq_size, output_size)
        logstds = self.logstds(self.relu(outputs))
        # sigmas = self.toCovariance(logstds) # (batch_size, output_size, output_size) or (batch_size, seq_size, output_size, output_size)
        return means, logstds.exp()


class RNN(Base):
    def __init__(self, num_predictors: int, hidden_size: int, n_layers: int, dropout: float, seed: int, last: bool = False) -> None:
        super(RNN, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed, last)
        kwargs = {'input_size': self.input_size, 'hidden_size': hidden_size, 'num_layers': n_layers, 'dropout': dropout, 'batch_first': True}
        self.model = self.functional()(**kwargs)
        self.embed = False

    def functional(self):
        return nn.RNN

    def mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.nn.utils.rnn.PackedSequence:
        ''' mask the output of the RNN '''
        # x (batch_size, seq_size, input_size)
        return pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
         
    def unmask(self, x: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        ''' unmask the output of the RNN '''
        output, _ = pad_packed_sequence(x, batch_first=True)
        return output

    def internal(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x (batch_size, seq_size, hidden_size)
        if self.last:
            x = self.mask(x, lengths) # type: ignore
        outputs, _ = self.model(x)
        if self.last:
            outputs = self.unmask(outputs) # (batch_size, max(lengths), hidden_size)
        return outputs


class GRU(RNN):
    def __init__(self, num_predictors: int, hidden_size: int, n_layers: int, dropout: float, seed: int, last: bool = False) -> None:
        super(GRU, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed, last)

    def functional(self): # type: ignore
        return nn.GRU


class LSTM(RNN):
    def __init__(self, num_predictors: int, hidden_size: int, n_layers: int, dropout: float, seed: int, last: bool = False) -> None:
        super(LSTM, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed, last)

    def functional(self): # type: ignore
        return nn.LSTM


class TransformerDecoder(Base):
    def __init__(self,
                 num_predictors: int,
                 hidden_size: int,
                 ff_size: int,
                 n_layers: int,
                 dropout: float,
                 seed: int,
                 last: bool = False,
                 n_heads: int = 4,
                 ) -> None:

        super(TransformerDecoder, self).__init__(num_predictors, hidden_size, n_layers, dropout, seed, last)
        decoder_layer = TransformerDecoderLayer(d_model=self.input_size,
                                                nhead=n_heads,
                                                dim_feedforward=ff_size,
                                                dropout=dropout,
                                                batch_first=True)
        self.model= nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.map2hidden = nn.Linear(self.input_size, hidden_size)

    def internal(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, seq_size, hidden_size)
        return self.model(x, x)


