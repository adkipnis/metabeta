import os
from tqdm import tqdm
from pathlib import Path
import argparse
import math
import torch
from tasks import FixedEffects


class Sower:
    ''' Seed generator for reproducibility. '''
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def throw(self) -> int:
        out = self.seed
        self.seed += 1
        return out


def getD(max_predictors: int) -> int:
    ''' Get a random number of predictors to draw from a linear model.'''
    shape = (1,)
    d = torch.randint(0, max_predictors + 1, shape)
    return int(d.item())


def getSigmaError(alpha: float = 3., beta: float = 1., clip: float = 1.5) -> float:
    ''' Get the noise standard deviation '''
    sigma_squared = torch.distributions.inverse_gamma.InverseGamma(alpha, beta).sample()
    sigma = math.sqrt(sigma_squared.item()) # type: ignore
    return min(sigma, clip)


def generateDataset(n_draws: int, max_samples: int, max_predictors: int, sower: Sower) -> dict:
    ''' Generate a dataset of linear model samples of varying length and width and return a DataLoader. '''
    data = []
    iterator = tqdm(range(n_draws))
    iterator.set_description(f'{part:02d}/{iterations:02d}')

    for _ in iterator:
        seed = sower.throw()
        torch.manual_seed(seed)
        d = getD(max_predictors)
        # sigma = math.sqrt(d + 1) * getSigmaError() if cfg.fixed == 0. else cfg.fixed
        sigma = getSigmaError() if cfg.fixed == 0. else cfg.fixed
        lm = FixedEffects(d, sigma, data_dist)
        data += [lm.sample(max_samples, seed, include_posterior=False)]
    return {'data': data, 'max_samples': max_samples, 'max_predictors': max_predictors}


def generateBalancedDataset(n_draws_per: int, max_samples: int, max_predictors: int) -> dict:
    ''' generateDataset but with balanced number of predictors for validation '''
    data = []
    iterator = tqdm(range(max_predictors + 1))
    iterator.set_description('Validation Set')
    sigmas = sorted([getSigmaError() for _ in range(n_draws_per)])

    for d in iterator:
        for seed in range(n_draws_per):
            torch.manual_seed(seed)
            # sigma = math.sqrt(d + 1) * float(sigmas[seed]) if cfg.fixed == 0. else cfg.fixed
            sigma = float(sigmas[seed]) if cfg.fixed == 0. else cfg.fixed
            lm = FixedEffects(d, sigma, data_dist)
            data += [lm.sample(max_samples, seed, include_posterior=True)]
    return {'data': data, 'max_samples': max_samples, 'max_predictors': max_predictors}


def dsFilename(size: int, part: int, suffix: str = '') -> Path:
    if size >= 1e6:
        n = f'{size/1e6:.0f}m'
    elif size >= 1e3:
        n = f'{size/1e3:.0f}k'
    else:
        n = str(size)
    p = f'{part:03d}'
    return Path('data', f'dataset-{n}-{p}{suffix}.pt')


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate datasets for linear model task.')
    parser.add_argument('-n', '--n_draws', type=int, default=int(1e4), help='Number of samples to draw per dataset (default = 10,000).')
    parser.add_argument('--n_draws_val', type=int, default=500, help='Number of samples (per d) to draw for validation dataset (default = 500).')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='Number of dataset partitions to generate (default = 100, 0 only generates validation dataset).')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum number of samples to draw per linear model (default = 50).')
    parser.add_argument('-d', '--max_predictors', type=int, default=14, help='Maximum number of predictors (without intercept) to draw per linear model (default = 14).')
    parser.add_argument('-b', '--begin', type=int, default=0, help='Begin with iteration number #b (default = 0).')
    parser.add_argument('-f', '--fixed', type=float, default=0., help='Fixed noise variance (default = 0. -> not fixed)')
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
    start = cfg.begin
    noise = "variable" if cfg.fixed == 0 else cfg.fixed
    suffix = f"-noise={noise}"
    data_dist = torch.distributions.uniform.Uniform(0., 1.)

    if start == 0:
        # generate validation dataset
        print(f'Generating validation dataset of {n_draws_val * (max_predictors + 1)} samples')
        dataset = generateBalancedDataset(n_draws_val, max_samples, max_predictors)
        filename = Path('data', f'dataset-val{suffix}.pt')
        torch.save(dataset, filename)
        start += 1
    else:
        # reset sower if starting from a different iteration
        seed = (start - 1) * n_draws + 1
        seed += n_draws_val * (max_predictors + 1)
        sower = Sower(seed)

    if iterations == 0:
        exit()

    # generate training datasets
    print(f'Generating {iterations} training datasets of {n_draws} samples each')
    for part in range(start, iterations + 1):
        dataset = generateDataset(n_draws, max_samples, max_predictors, sower)
        filename = dsFilename(n_draws, part, suffix)
        torch.save(dataset, filename)
        print(f'Saved dataset to {filename}')

