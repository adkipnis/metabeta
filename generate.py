import os
from tqdm import tqdm
from pathlib import Path
import torch
from tasks import Task, LinearModel
from dataset import RnnDataset


class Sower:
    ''' Seed generator for reproducibility. '''
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def throw(self) -> int:
        out = self.seed
        self.seed += 1
        return out


def getD(seed: int, max_predictors: int) -> int:
    ''' Get a random number of predictors to draw from a linear model.'''
    torch.manual_seed(seed)
    d = torch.randint(0, max_predictors, (1,))
    return int(d.item())


def getN(seed:int, n_predictors: int, max_samples: int) -> int:
    ''' Get a random number of samples to draw from a linear model, depending on the number of predictors.'''
    torch.manual_seed(seed)
    low = 5 * (n_predictors + 1)
    n = torch.randint(low, max_samples, (1,))
    return int(n.item())


