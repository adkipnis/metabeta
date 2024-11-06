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


def getSigmaError(seed: int, alpha: float = 2., beta: float = 1., eps: float = 1e-6) -> float:
    ''' Get a random error term for a linear model.'''
    torch.manual_seed(seed)
    gamma = torch.distributions.gamma.Gamma(alpha, beta).sample()
    sigma = 1. / (gamma + eps)
    return sigma.item()


def getDataDist(seed: int) -> torch.distributions.Distribution:
    ''' Get a random distribution for the data.'''
    torch.manual_seed(seed)
    if torch.rand(1) > 0.5:
        return torch.distributions.uniform.Uniform(-5., 5.)
    else:
        return torch.distributions.normal.Normal(0., 3.)


def generateDataset(n_draws: int, max_samples: int, max_predictors: int, sower: Sower) -> RnnDataset:
    ''' Generate a dataset of linear model samples of varying length and width and return a DataLoader. '''
    samples = []
    for _ in tqdm(range(n_draws)):
        seed = sower.throw()
        d = getD(seed, max_predictors)
        n = max_samples # alternatively: getN(seed, d, max_samples)
        sigma_error = getSigmaError(seed)
        data_dist = getDataDist(seed)
        task = Task(d, seed, sigma_error)
        lm = LinearModel(task, data_dist)
        samples += [lm.sample(n, seed)]
    return RnnDataset(samples, max_samples, max_predictors)


