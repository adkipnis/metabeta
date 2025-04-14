import os
import itertools
from tqdm import tqdm
import argparse
import torch
from torch import distributions as D
from tasks import FixedEffects, MixedEffects
from utils import dsFilename


def sampleInt(batch_size: int, min_val: int, max_val: int) -> torch.Tensor:
    ''' sample batch from discrete uniform (include the max_val as possible value) '''
    return torch.randint(min_val, max_val+1, (batch_size,))

def sampleHN(shape: tuple, scale: float, max_val: float = float('inf'),) -> torch.Tensor:
    ''' sample batch from halfnormal, optionally clamp '''
    val = D.HalfNormal(scale).sample(shape)
    if max_val < float('inf'):
        val = val.clamp(max=max_val) # type: ignore
    return val # type: ignore

def sampleIG(shape: tuple, alpha: float, beta: float, max_val: float = float('inf')) -> torch.Tensor:
    ''' sample batch from inverse gamma (and apply sqrt), optionally clamp '''
    val = D.InverseGamma(alpha, beta).sample(shape).sqrt() # type: ignore
    if max_val < float('inf'):
        val = val.clamp(max=max_val) # type: ignore
    return val # type: ignore


def constrain(a: torch.Tensor, b: torch.Tensor, max_ratio: float, min_val: int):
    ''' constrain b such that a * max_ratio >= b (element-wise) '''
    bounds = a * max_ratio
    indices = bounds < b
    b[indices] = bounds[indices].floor().to(b.dtype).clamp(min=min_val)
    return a, b


    data = []
    iterator = tqdm(range(n_draws))
    iterator.set_description(f'{part:02d}/{iterations:02d}')
    LinearModel = FixedEffects if ds_type == "ffx" else MixedEffects

    for _ in iterator:
        seed = sower.throw()
        torch.manual_seed(seed)
        d = getD(0, max_predictors)
        q = getD(0, d//2) if ds_type == "mfx" else 0
        m = 5 if ds_type == "mfx" else 1 # todo: vary number of groups
        sigma = ufNoise() if cfg.fixed == 0. else cfg.fixed
        if cfg.scale_noise:
            sigma = sigma * sqrt(d + 1)
        lm = LinearModel(d, sigma, data_dist, q, m)
        data += [lm.sample(max_samples, seed, include_posterior=False)]
    return {'data': data, 'max_samples': max_samples, 'max_predictors': max_predictors}


def generateBalancedDataset(ds_type: str, n_draws_per: int, max_samples: int, max_predictors: int) -> Tuple[dict, list]:
    ''' generateDataset but with balanced number of predictors for validation '''
    data = []
    iterator = tqdm(range(max_predictors + 1))
    iterator.set_description('Validation Set')
    sigmas = torch.linspace(0.05, 1.5, steps=n_draws_per)
    LinearModel = FixedEffects if ds_type == "ffx" else MixedEffects
    info = []

    for d in iterator:
        q_range = range(d//2 + 1) if ds_type == "mfx" else range(1)
        for q in q_range:
            for i in range(n_draws_per):
                torch.manual_seed(i)
                sigma = sigmas[i] if cfg.fixed == 0. else cfg.fixed
                m = 5 if ds_type == "mfx" else 1 # todo: vary number of groups
                if cfg.scale_noise:
                    sigma = sigma * sqrt(d + 1)
                lm = LinearModel(d, sigma.item(), data_dist, q, m)
                data += [lm.sample(max_samples, i, include_posterior=False)]
                info += [{'d': d, 'q': q, 'sigma': sigma.item()}]
    return {'data': data, 'max_samples': max_samples, 'max_predictors': max_predictors}, info


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
    parser.add_argument('-s', '--scale_noise', action='store_true', help='scale noise with number of predictors (default = false)')
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
    data_dist = D.Uniform(0., 1.)

    if start == 0:
        # generate validation dataset
        print(f'Generating {ds_type} validation dataset...')
        dataset, info = generateBalancedDataset(ds_type, n_draws_val, max_samples, max_predictors)
        print(f'Number of samples: {len(dataset["data"])}')
        filename = dsFilenameVal(ds_type, max_predictors, max_samples, fixed)
        filename_info = dsFilenameVal(ds_type, max_predictors, max_samples, fixed, '_info')
        torch.save(dataset, filename)
        torch.save(info, filename_info)
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

