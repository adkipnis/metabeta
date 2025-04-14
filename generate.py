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


def genFfxTrainSet(batch_size: int, seed: int, include_posterior: bool = False) -> dict:
    ''' Generate a [batch_size] fixed effects model datasets with varying n and d. '''
    # init
    torch.manual_seed(seed)
    data = []
    iterator = tqdm(range(batch_size))
    iterator.set_description(f'{part:02d}/{cfg.iterations:02d}')

    # presample hyperparams
    sigma = sampleHN((batch_size,), 1., cfg.max_sigma)
    d = sampleInt(batch_size, 0, cfg.max_d).tolist()
    n = sampleInt(batch_size, cfg.max_d, cfg.max_n).tolist()

    # sample datasets
    for i in iterator:
        lm = FixedEffects(sigma[i], d[i])
        data += [lm.sample(n[i], include_posterior=include_posterior)]
    info = {'d': d, 'n': n, 'sigma': sigma,
            'max_d': cfg.max_d, 'max_n': cfg.max_n, 'max_sigma': cfg.max_sigma,
            'seed': seed}
    return {'data': data, 'info': info}

def genFfxValSet(repeats: int, seed: int, include_posterior: bool = False) -> dict:
    ''' Generate a [batch_size] fixed effects model datasets with varying n and d. '''
    # init
    torch.manual_seed(seed)
    data = []

    # presample hyperparams
    d_full = [0, 1, 2, 4, 8]
    n_full = [50, 30, 15]
    sigma_full = D.HalfNormal(1.).icdf(torch.tensor([0.01, 0.25, 0.5, 0.75])).tolist()
    combinations = list(itertools.product(d_full, n_full, sigma_full)) * repeats
    d, n, sigma = zip(*combinations)

    # sample datasets
    for i in range(len(combinations)):
        d_, n_, s_ = combinations[i]
        lm = FixedEffects(s_, d_)
        data += [lm.sample(n_, include_posterior=include_posterior)]
    info = {'d': list(d), 'n': list(n), 'sigma': torch.tensor(sigma),
            'max_d': cfg.max_d, 'max_n': cfg.max_n, 'max_sigma': cfg.max_sigma,
            'seed': seed}
    return {'data': data, 'info': info}

def genMfxTrainSet(batch_size: int, seed: int, include_posterior: bool = False) -> dict:
    ''' Generate a [batch_size] mixed effects model datasets with varying n, m, d, q. '''
    # init
    torch.manual_seed(seed)
    data = []
    iterator = tqdm(range(batch_size))
    iterator.set_description(f'{part:02d}/{cfg.iterations:02d}')

    # presample hyperparams
    sigma = sampleHN((batch_size, cfg.max_d//2 + 1), 1., cfg.max_sigma)
    d = sampleInt(batch_size, 0, cfg.max_d)
    n = sampleInt(batch_size, cfg.max_d, cfg.max_n)
    q = sampleInt(batch_size, 0, cfg.max_d//2)
    m = sampleInt(batch_size, 2, cfg.max_n//10)
    
    # apply constraints
    d, q = constrain(d, q, 0.5, min_val=0)
    n, m = constrain(n, m, 0.1, min_val=2)
    m[q==0] = 1
    d, q, n, m = d.tolist(), q.tolist(), n.tolist(), m.tolist()
    
    # sample datasets
    for i in iterator:
        lm = MixedEffects(sigma[i,0], sigma[i,1:], d[i], q[i], m[i])
        data += [lm.sample(n[i], include_posterior=include_posterior)]
    info = {'d': d, 'q': q, 'n': n, 'm': m, 'sigma': sigma,
            'max_d': cfg.max_d, 'max_n': cfg.max_n, 'max_sigma': cfg.max_sigma,
            'seed': seed}
    return {'data': data, 'info': info}


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate datasets for linear model task.')
    parser.add_argument('-t', '--type', type=str, default='ffx', help='Type of dataset [ffx, mfx] (default = mfx)')
    parser.add_argument('--bs_train', type=int, default=int(1e4), help='batch size per training partition (default = 10,000).')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Number of dataset partitions to generate (default = 10, 0 only generates validation dataset).')
    parser.add_argument('-n', '--max_n', type=int, default=50, help='Maximum number of samples to draw per linear model (default = 50).')
    parser.add_argument('-d', '--max_d', type=int, default=8, help='Maximum number of predictors (without intercept) to draw per linear model (default = 8).')
    parser.add_argument('--max_sigma', type=float, default=float('inf'), help='Maximum value for a sampled standard deviation')
    parser.add_argument('-b', '--begin', type=int, default=0, help='Begin with iteration number #b (default = 0).')
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

