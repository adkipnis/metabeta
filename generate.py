import os
from tqdm import tqdm
from pathlib import Path
import argparse
import math
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


def getSigmaError(seed: int, nu: float = 3., clip: float = 2.) -> float:
    ''' Sample the noise standard deviation from the inverse Chi-Squared distribution '''
    torch.manual_seed(seed)
    sigma_squared_inv = torch.distributions.chi2.Chi2(nu).sample()
    sigma_squared = 1. / (sigma_squared_inv + 1e-6)
    sigma = math.sqrt(sigma_squared.item()) # type: ignore
    return min(sigma, clip) 


# def getSigmaError(seed: int, alpha: float = 3., beta: float = 1., clip: float = 2.) -> float:
#     ''' Get the noise standard deviation '''
#     torch.manual_seed(seed)
#     sigma_squared = torch.distributions.inverse_gamma.InverseGamma(alpha, beta).sample()
#     sigma = math.sqrt(sigma_squared.item()) # type: ignore
#     return min(sigma, clip) 


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
        sigma = getSigmaError(seed)
        data_dist = getDataDist(seed)
        lm = FixedEffects(d, sigma, data_dist)
        samples += [lm.sample(max_samples, seed, include_posterior=False)]
    return LMDataset(samples, max_samples, max_predictors)


def generateBalancedDataset(n_draws_per: int, max_samples: int, max_predictors: int, sower: Sower) -> LMDataset:
    ''' generateDataset but with balanced number of predictors for validation '''
    samples = []
    d = 0 # for iterator description
    iterator = tqdm(range(max_predictors + 1))
    iterator.set_description(f'{d:02d}/{max_predictors:02d}')

    # generate sigmas in a balanced way
    # x = torch.arange(0., 3., step=3./n_draws_per)
    # sigmas = 3. * torch.exp(x - 3.)
    sigmas = sorted([getSigmaError(i) for i in range(n_draws_per)])

    for d in iterator:
        for seed in range(n_draws_per):
            sigma = float(sigmas[seed])
            data_dist = getDataDist(seed)
            lm = FixedEffects(d, sigma, data_dist)
            samples += [lm.sample(max_samples, seed, include_posterior=True)]
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
    parser.add_argument('--n_draws_val', type=int, default=500, help='Number of samples (per d) to draw for validation dataset (default = 500).')
    parser.add_argument('-i', '--iterations', type=int, default=0, help='Number of dataset partitions to generate (default = 0, only generates validatioin dataset).')
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
        filename = Path('data', 'dataset-val.pt')
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
        filename = dsFilename(n_draws, part)
        torch.save(dataset, filename)
        print(f'Saved dataset to {filename}')

