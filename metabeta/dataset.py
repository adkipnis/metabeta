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


