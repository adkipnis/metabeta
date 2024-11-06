from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class BaseRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int) -> None:
        super(BaseRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.seed = seed
        self.initializeWeights()
    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        # Pack the padded sequence
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # type: ignore
        
