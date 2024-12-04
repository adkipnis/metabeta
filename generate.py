import os
from tqdm import tqdm
from pathlib import Path
import argparse
import torch
from tasks import FixedEffects
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
    d = torch.randint(0, max_predictors + 1, (1,))
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
        sigma = 0.1 # getSigmaError(seed)
        data_dist = getDataDist(seed)
        lm = FixedEffects(d, sigma, data_dist)
        samples += [lm.sample(max_samples, seed)]
    return LMDataset(samples, max_samples, max_predictors)


def generateBalancedDataset(n_draws_per: int, max_samples: int, max_predictors: int, sower: Sower) -> LMDataset:
    ''' generateDataset but with balanced number of predictors for validation '''
    samples = []
    d = 0 # for iterator description
    iterator = tqdm(range(max_predictors + 1))
    iterator.set_description(f'{d:02d}/{max_predictors:02d}')
    for d in iterator:
        for _ in range(n_draws_per):
            seed = sower.throw()
            sigma = 0.1
            data_dist = getDataDist(seed)
            lm = FixedEffects(d, sigma, data_dist)
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


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate datasets for linear model task.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for random number generator (default = 0).')
    parser.add_argument('-n', '--n_draws', type=int, default=int(1e4), help='Number of samples to draw per dataset (default = 10,000).')
    parser.add_argument('--n_draws_val', type=int, default=200, help='Number of samples (per d) to draw for validation dataset (default = 200).')
    parser.add_argument('-i', '--iterations', type=int, default=int(1e3), help='Number of dataset partitions to generate (default = 1,000).')
    parser.add_argument('--max_samples', type=int, default=200, help='Maximum number of samples to draw per linear model (default = 200).')
    parser.add_argument('-d', '--max_predictors', type=int, default=14, help='Maximum number of predictors (without intercept) to draw per linear model (default = 14).')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration number (default = 1).')
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    cfg = setup()
    sower = Sower(cfg.seed)
    n_draws = cfg.n_draws
    n_draws_val = cfg.n_draws_val
    iterations = cfg.iterations
    max_samples = cfg.max_samples
    max_predictors = cfg.max_predictors
    start = cfg.start

    if start == 1:
        # generate validation dataset
        print(f'Generating validation dataset of {n_draws_val * (max_predictors + 1)} samples')
        dataset = generateBalancedDataset(n_draws_val, max_samples, max_predictors, sower)
        filename = Path('data', 'dataset-val-fixed-sigma.pt')
        torch.save(dataset, filename)
    else:
        # reset sower if starting from a different iteration
        seed = (start - 1) * n_draws + 1
        seed += n_draws_val * (max_predictors + 1)
        sower = Sower(seed)

    # generate training datasets
    print(f'Generating {iterations} training datasets of {n_draws} samples each')
    for part in range(start, iterations + 1):
        dataset = generateDataset(n_draws, max_samples, max_predictors, sower)
        filename = dsFilename(n_draws, part, "fixed-sigma")
        torch.save(dataset, filename)
        print(f'Saved dataset to {filename}')

