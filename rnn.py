from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoderLayer

class BaseRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(BaseRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.means = nn.Linear(hidden_size, output_size)
        self.logstds = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.seed = seed
        self.reuse = reuse
        self.initializeWeights()

    def initializeWeights(self) -> None:
        seed = self.seed
        for p in self.parameters():
            if p.dim() > 1:
                torch.manual_seed(seed)
                seed += 1
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pack the padded sequence
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # type: ignore
        
        # Forward pass through GRU
        packed_outputs, _ = self.rnn(packed_input)

        # back to tensor
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        if not self.reuse:
            outputs = outputs[:, -1]
        
        # transform outputs
        means = self.means(self.relu(outputs))
        logstds = self.logstds(self.relu(outputs))
        
        return means, logstds


class GRU(BaseRNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(GRU, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)


class LSTM(BaseRNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(LSTM, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)


class TransformerDecoder(BaseRNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True, nhead: int = 4, num_layers: int = 1) -> None:
        super(TransformerDecoder, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        decoder_layer = TransformerDecoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create mask for padding
        mask = self.create_mask(x, lengths)

        # Forward pass through TransformerDecoder
        outputs = self.transformer_decoder(x, x, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        
        # Linear transformation
        outputs = self.output_linear(outputs)

        if not self.reuse:
            outputs = outputs[:, -1]
        
        # Transform outputs
        means = self.means(self.relu(outputs))
        logstds = self.logstds(self.relu(outputs))
        
        return means, logstds

    def create_mask(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        return mask


