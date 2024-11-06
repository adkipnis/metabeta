from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaseRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int) -> None:
        super(BaseRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.means = nn.Linear(hidden_size, output_size)
        self.logstds = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.seed = seed
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
        
        # transform outputs
        means = self.means(self.relu(outputs))
        logstds = self.logstds(self.relu(outputs))
        
        return means, logstds


class GRU(BaseRNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int) -> None:
        super(GRU, self).__init__(input_size, hidden_size, output_size, seed)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)


class LSTM(BaseRNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int) -> None:
        super(LSTM, self).__init__(input_size, hidden_size, output_size, seed)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

