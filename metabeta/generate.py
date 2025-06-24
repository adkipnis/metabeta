import os
from sys import exit
from tqdm import tqdm
import argparse
import torch
from torch import distributions as D

from metabeta.tasks import FixedEffects, MixedEffects
from metabeta.utils import dsFilename
from metabeta.markov import fitFFX, fitMFX


# -----------------------------------------------------------------------------
# helpers
def sampleInt(batch_size: int | tuple, min_val: int, max_val: int) -> torch.Tensor:
    ''' sample batch from discrete uniform (include the max_val as possible value) '''
    shape = (batch_size,) if isinstance(batch_size, int) else batch_size
    return torch.randint(min_val, max_val+1, shape)

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


# -----------------------------------------------------------------------------
# FFX
def genFFX(batch_size: int, seed: int, mcmc: bool = False, analytical: bool = True) -> dict:
    ''' Generate a [batch_size] fixed effects model datasets with varying n and d. '''
    # init
    torch.manual_seed(seed)
    data = []
    iterator = tqdm(range(batch_size))
    iterator.set_description(f'{part:02d}/{cfg.iterations:02d}')

    # presample hyperparams
    sigma = sampleHN((batch_size,), 1., cfg.max_sigma)
    d = sampleInt(batch_size, cfg.min_d, cfg.max_d).tolist()
    n = sampleInt(batch_size, cfg.min_n, cfg.max_n).tolist()

    # sample datasets
    for i in iterator:
        lm = FixedEffects(sigma[i], d[i])
        ds = lm.sample(n[i], include_posterior=analytical)
        if mcmc:
            ds['mcmc'] = fitFFX(**ds)
        data += [ds]
    info = {'d': d, 'n': n, 'sigma': sigma,
            'max_d': cfg.max_d, 'max_n': cfg.max_n,
            'max_q': 0, 'max_m': 0,
            'max_sigma': cfg.max_sigma,
            'seed': seed}
    return {'data': data, 'info': info}


# -----------------------------------------------------------------------------
# MFX

def genMFX(batch_size: int, seed: int, mcmc: bool = False, analytical: bool = False) -> dict:
    ''' Generate a [batch_size] mixed effects model datasets with varying n, m, d, q. '''
    # init
    torch.manual_seed(seed)
    data = []
    iterator = tqdm(range(batch_size))
    iterator.set_description(f'{part:02d}/{cfg.iterations:02d}')

    # presample hyperparams
    min_q = max_q = 1
    sigma = sampleHN((batch_size, max_q + 1), 1., cfg.max_sigma)
    d_ = sampleInt(batch_size, cfg.max_d, cfg.max_d) + 1
    m = sampleInt(batch_size, cfg.min_m, cfg.max_m)
    n = sampleInt((batch_size, cfg.max_m), cfg.min_n, cfg.max_n)
    q = sampleInt(batch_size, min_q, max_q)
    
    # apply constraints
    m, d_ = constrain(m, d_, 0.5, min_val=1) # at least two subjects per covariate
    d_, q = constrain(d_, q, 0.5, min_val=min_q) # at least two fixed effects per random effect
    d = d_ - 1
    
    d, m, n, q = d.tolist(), m.tolist(), n.tolist(), q.tolist()
    
    # sample datasets
    for i in iterator:
        sigma_e = sigma[i,0]
        sigmas_rfx = sigma[i,1:][:q[i]]
        lm = MixedEffects(sigma_e, sigmas_rfx, d[i], q[i], m[i], n[i][:m[i]])
        ds = lm.sample()
        if mcmc:
            ds['mcmc'] = fitMFX(**ds)
        data += [ds]
    info = {'d': d, 'q': q, 'n': n, 'm': m, 'sigma': sigma,
            'max_d': cfg.max_d, 'max_n': cfg.max_n,
            'max_q': max(q), 'max_m': cfg.max_m,
            'max_sigma': cfg.max_sigma,
            'seed': seed}
    return {'data': data, 'info': info}



# =============================================================================
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate datasets for linear model task.')
    parser.add_argument('-t', '--type', type=str, default='ffx', help='Type of dataset [ffx, mfx] (default = mfx)')
    parser.add_argument('--bs_train', type=int, default=int(10e3), help='batch size per training partition (default = 50,000).')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='Number of dataset partitions to generate (default = 50, 0 only generates validation dataset).')
    parser.add_argument('--max_m', type=int, default=30, help='MFX: Maximum number of groups (default = 30).')
    parser.add_argument('--min_m', type=int, default=5, help='MFX: Minimum number of groups (default = 5).')
    parser.add_argument('--max_n', type=int, default=50, help='Maximum number of samples per group (default = 20).')
    parser.add_argument('--min_n', type=int, default=30, help='Minimum number of samples per group (default = 5).')
    parser.add_argument('--min_d', type=int, default=1, help='Minimum number of predictors (without intercept) to draw per linear model (default = 0).')
    parser.add_argument('--max_d', type=int, default=1, help='Maximum number of predictors (without intercept) to draw per linear model (default = 3).')
    parser.add_argument('--max_sigma', type=float, default=float('inf'), help='Maximum value for a sampled standard deviation')
    parser.add_argument('-b', '--begin', type=int, default=51, help='Begin with iteration number #b (default = 0).')
    return parser.parse_args()


