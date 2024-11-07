from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoderLayer

class Base(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True, pack: bool = False) -> None:
        super(Base, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.means = nn.Linear(hidden_size, output_size)
        self.logstds = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.seed = seed
        self.reuse = reuse
        self.pack = pack
        self.initializeWeights()

    def initializeWeights(self) -> None:
        seed = self.seed
        for p in self.parameters():
            if p.dim() > 1:
                torch.manual_seed(seed)
                seed += 1
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, lengths: Union[None, List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # optionally pack the padded sequence
        if self.pack and isinstance(lengths, list):
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # type: ignore

        # main forward pass
        outputs, _ = self.model(x) # (batch_size, seq_len, hidden_size)

        # optionally unpack the sequence
        if self.pack:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # optionally select the last output
        if not self.reuse:
            outputs = outputs[:, -1]

        # transform outputs
        means = self.means(self.relu(outputs))
        logstds = self.logstds(self.relu(outputs))
        return means, logstds


class GRU(Base):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(GRU, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)


class LSTM(Base):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True) -> None:
        super(LSTM, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)


class TransformerDecoder(Base):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int, reuse: bool = True, nhead: int = 4, num_layers: int = 1) -> None:
        super(TransformerDecoder, self).__init__(input_size, hidden_size, output_size, seed, reuse)
        decoder_layer = TransformerDecoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, lengths: Union[None, List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Optionally mask the padded sequence and forward pass through TransformerDecoder
        if self.pack and isinstance(lengths, list):
            mask = self.paddingMask(x, lengths)
            outputs = self.transformer_decoder(x, x, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        else:
            outputs = self.transformer_decoder(x, x) # (batch_size, seq_len, hidden_size)

        # Forward pass through TransformerDecoder
        if not self.reuse:
            outputs = outputs[:, -1]
        
        # Linear transformation
        outputs = self.output_linear(outputs)
        
        # Transform outputs
        means = self.means(self.relu(outputs))
        logstds = self.logstds(self.relu(outputs))
        return means, logstds


    def paddingMask(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        return mask

