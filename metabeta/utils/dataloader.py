from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from metabeta.utils.sampling import samplePermutation
from metabeta.utils.padding import unpad


class Collection(Dataset):
    def __init__(
        self,
        path: Path,
        permute: bool = True,
    ):
        super().__init__()

