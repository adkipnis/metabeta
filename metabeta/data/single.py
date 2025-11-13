# generate a single hierarchical dataset
from dataclasses import dataclass
import torch
from torch import distributions as D
from metabeta.utils import fullCovary


# -----------------------------------------------------------------------------
def standardize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x = x - x.mean(dim, keepdim=True)
    x = x / x.std(dim, keepdim=True)
    return x

def standardnormal(*size):
    x = torch.randn(size)
    return standardize(x)

