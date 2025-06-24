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


