import os
from tqdm import tqdm
import argparse
from math import sqrt
import torch
from torch import distributions as D
from tasks import FixedEffects, MixedEffects
from utils import dsFilename, dsFilenameVal


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


def ufNoise(min_val: float = 0.05, max_val: float = 1.5) -> float:
    ''' Get the noise standard deviation '''
    sigma = D.Uniform(min_val, max_val).sample()
    return sigma.item()


def igNoise(alpha: float = 3., beta: float = 1., max_val: float = 1.5) -> float:
    ''' Get the noise standard deviation '''
    sigma_squared = D.InverseGamma(alpha, beta).sample()
    sigma = sigma_squared.sqrt()
    sigma = sigma.clamp(max=max_val)
    return sigma.item()


def generateDataset(ds_type: str, n_draws: int, max_samples: int, max_predictors: int, sower: Sower) -> dict:
    ''' Generate a dataset of linear model samples of varying length and width and return a DataLoader. '''
    data = []
    iterator = tqdm(range(n_draws))
    iterator.set_description(f'{part:02d}/{iterations:02d}')
    LinearModel = FixedEffects if ds_type == "ffx" else MixedEffects

    for _ in iterator:
        seed = sower.throw()
        torch.manual_seed(seed)
        d = getD(max_predictors)
        sigma = ufNoise() if cfg.fixed == 0. else cfg.fixed
        if cfg.scale_noise:
            sigma = sigma * sqrt(d + 1)
        lm = LinearModel(d, sigma, data_dist)
        data += [lm.sample(max_samples, seed, include_posterior=False)]
    return {'data': data, 'max_samples': max_samples, 'max_predictors': max_predictors}


def generateBalancedDataset(ds_type: str, n_draws_per: int, max_samples: int, max_predictors: int) -> dict:
    ''' generateDataset but with balanced number of predictors for validation '''
    data = []
    iterator = tqdm(range(max_predictors + 1))
    iterator.set_description('Validation Set')
    sigmas = [ufNoise() for _ in range(n_draws_per)]
    LinearModel = FixedEffects if ds_type == "ffx" else MixedEffects
    include_posterior = ds_type == "ffx"

    for d in iterator:
        for i in range(n_draws_per):
            torch.manual_seed(i)
            sigma = sigmas[i] if cfg.fixed == 0. else cfg.fixed
            if cfg.scale_noise:
                sigma = sigma * sqrt(d + 1)
            lm = LinearModel(d, sigma, data_dist)
            data += [lm.sample(max_samples, i, include_posterior=include_posterior)]
    return {'data': data, 'max_samples': max_samples, 'max_predictors': max_predictors}


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate datasets for linear model task.')
    parser.add_argument('-t', '--type', type=str, default='mfx', help='Type of dataset [ffx, mfx] (default = mfx)')
    parser.add_argument('--n_draws', type=int, default=int(1e4), help='Number of samples to draw per dataset (default = 10,000).')
    parser.add_argument('--n_draws_val', type=int, default=500, help='Number of samples (per d) to draw for validation dataset (default = 500).')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Number of dataset partitions to generate (default = 10, 0 only generates validation dataset).')
    parser.add_argument('-n', '--max_samples', type=int, default=50, help='Maximum number of samples to draw per linear model (default = 50).')
    parser.add_argument('-d', '--max_predictors', type=int, default=8, help='Maximum number of predictors (without intercept) to draw per linear model (default = 8).')
    parser.add_argument('-b', '--begin', type=int, default=0, help='Begin with iteration number #b (default = 0).')
    parser.add_argument('-f', '--fixed', type=float, default=0., help='Fixed noise variance (default = 0. -> not fixed)')
    parser.add_argument('-s', '--scale_noise', action='store_false', help='scale noise with number of predictors')
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    cfg = setup()
    ds_type = cfg.type
    n_draws = cfg.n_draws
    n_draws_val = cfg.n_draws_val
    iterations = cfg.iterations
    max_samples = cfg.max_samples
    max_predictors = cfg.max_predictors
    fixed = cfg.fixed
    start = cfg.begin
    seed = n_draws_val + 1
    data_dist = torch.distributions.uniform.Uniform(0., 1.)

    if start == 0:
        # generate validation dataset
        print(f'Generating {ds_type} validation dataset of {n_draws_val * (max_predictors + 1)} samples')
        dataset = generateBalancedDataset(ds_type, n_draws_val, max_samples, max_predictors)
        filename = dsFilenameVal(ds_type, max_predictors, max_samples, fixed)
        torch.save(dataset, filename)
        start += 1
    else:
        seed += (start - 1) * n_draws

    if iterations == 0:
        exit()

    # generate training datasets
    print(f'Generating {iterations} training datasets of {n_draws} samples each')
    sower = Sower(seed)
    for part in range(start, iterations + 1):
        dataset = generateDataset(ds_type, n_draws, max_samples, max_predictors, sower)
        filename = dsFilename(ds_type, max_predictors, max_samples, fixed, n_draws, part)
        torch.save(dataset, filename)
        print(f'Saved dataset to {filename}')

