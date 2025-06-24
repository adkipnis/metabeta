from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def padTensor(tensor: torch.Tensor, shape: tuple, value=0) -> torch.Tensor:
    assert len(tensor.shape) == len(shape),\
        "Input tensor and target shape must have the same number of dimensions"
    pad_size = [max(s - t, 0) for s, t in zip(reversed(shape), reversed(tensor.shape))]
    padding = []
    for p in pad_size:
        padding.extend([0, p])  # Pad at the end of each dimension
    return F.pad(tensor, padding, value=value)


class LMDataset(Dataset):
    def __init__(self,
                 data: List[dict],
                 info: dict,
                 permute: bool = True,
                 ) -> None:
        self.info = info
        self.permute = permute
        self.n_max = self.info['max_n']
        self.d_max = self.info['max_d']
        self.m_max = self.info['max_m']
        self.q_max = self.info['max_q']
        self.len_max = self.n_max * self.m_max
        
        self.fx_type = 'mfx' if 'rfx' in data[0] else 'ffx'
        self.data = [self.preprocess(item) for item in data]

    def __len__(self) -> int:
       return len(self.data)
   
    def __getitem__(self, idx) -> dict:
        out = self.data[idx]
        return out

    def getPermutation(self, d: int):
        idx = torch.randperm(d-1) + 1
        zero = torch.zeros((1,), dtype=idx.dtype)
        idx = torch.cat([zero, idx])
        return idx

    def preprocess(self, item: dict) -> dict:
       if self.fx_type == 'ffx':
           return self.preprocessFfx(item)
       elif self.fx_type == 'mfx':
           return self.preprocessMfx(item)
       else:
           raise ValueError(f'Unsupported fx type {self.fx_type} for preprocessing')


