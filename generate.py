import os
from tqdm import tqdm
from pathlib import Path
import torch
from tasks import Task, LinearModel
from dataset import LMDataset


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


def getSigmaError(seed: int, alpha: float = 2.75, beta: float = 1., clip: float = 3., eps: float = 1e-6) -> float:
    ''' Get a random error term for a linear model.'''
    torch.manual_seed(seed)
    gamma = torch.distributions.gamma.Gamma(alpha, beta).sample()
    sigma = 1. / (gamma + eps)
    return min(sigma.item(), clip)


def getDataDist(seed: int) -> torch.distributions.Distribution:
    ''' Get a random distribution for the data.'''
    torch.manual_seed(seed)
    if torch.rand(1) > 0.5:
        return torch.distributions.uniform.Uniform(-5., 5.)
    else:
        return torch.distributions.normal.Normal(0., 3.)


def generateDataset(n_draws: int, max_samples: int, max_predictors: int, sower: Sower) -> LMDataset:
    ''' Generate a dataset of linear model samples of varying length and width and return a DataLoader. '''
    samples = []
    iterator = tqdm(range(n_draws))
    iterator.set_description(f'{part:02d}/{iterations:02d}')
    for _ in iterator:
        seed = sower.throw()
        d = getD(seed, max_predictors)
        sigma_error = 0.1 # alternatively: getSigmaError(seed)
        data_dist = getDataDist(seed)
        task = Task(d, seed, sigma_error)
        lm = LinearModel(task, data_dist)
        samples += [lm.sample(max_samples, seed)]
    return LMDataset(samples, max_samples, max_predictors)


def dsFilename(size: int, part: int, suffix: str = '') -> Path:
    if size >= 1e6:
        n = f'{size/1e6:.0f}m'
    elif size >= 1e3:
        n = f'{size/1e3:.0f}k'
    else:
        n = str(size)
    p = f'{part:03d}'
    if suffix:
        suffix = '-' + suffix
    return Path('data', f'dataset-{n}-{p}{suffix}.pt')


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    sower = Sower(0)
    n_draws = int(1e4)
    iterations = 500
    max_samples = 200
    max_predictors = 18
    start = 1

    # reset sower if starting from a different iteration
    if start > 1:
        seed = (start - 1) * n_draws + 1
        sower = Sower(seed)

    # generate datasets
    for part in range(start, iterations + 1):
        dataset = generateDataset(n_draws, max_samples, max_predictors, sower)
        filename = dsFilename(n_draws, part, "fixed-sigma")
        torch.save(dataset, filename)
        print(f'Saved dataset to {filename}')

