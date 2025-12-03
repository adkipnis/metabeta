from sys import exit
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch import distributions as D

from metabeta.utils import dsFilename, padTensor
from metabeta.data.single import Prior, Synthesizer, Emulator, Generator
from metabeta.data.fit import fitPyMC, fitBambi


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hierarchical datasets.')
    parser.add_argument('--bs_train', type=int, default=4096, help='batch size per training partition (default = 4,096).')
    parser.add_argument('--bs_val', type=int, default=256, help='batch size for validation partition (default = 256).')
    parser.add_argument('--bs_test', type=int, default=256, help='batch size per testing partition (default = 256).')
    parser.add_argument('--bs_load', type=int, default=16, help='Batch size when loading (for grouping n, default = 16)')
    parser.add_argument('--min_m', type=int, default=5, help='Minimum number of groups (default = 5).')
    parser.add_argument('-m', '--max_m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='Minimum number of samples per group (default = 10).')
    parser.add_argument('-n', '--max_n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    parser.add_argument('-d', '--max_d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes) to draw per linear model.')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='Maximum number of random effects (intercept + slopes) to draw per linear model.')
    parser.add_argument('--d_tag', type=str, default='all', help='Suffix for model ID (default = '')')
    parser.add_argument('--api', type=str, default='bambi', help='API to use for competetive fit (default = "bambi")')
    parser.add_argument('--toy', action='store_true', help='Generate toy data (default = False)')
    parser.add_argument('--semi', action='store_true', help='Generate semi-synthetic data (default = False)')
    parser.add_argument('--sub', action='store_true', help='Generate sub-sampled real data (default = False)')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD for semi-synthetic data (default = False)')
    parser.add_argument('--cache', action='store_false', help='Save each fitted dataset separately to cache (default = True)')
    parser.add_argument('--slurm', action='store_true', help='Turn off multiprocessing for slurm (default = False)')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Seed for PyMC fit (default = 42)')
    parser.add_argument('-b', '--begin', type=int, default=0, help='Begin with iteration number #b.')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Number of dataset partitions to generate.')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# helpers
def getFileName(ds_type: str, part: int, suffix: str = '') -> Path:
    if 'train' in ds_type:
        size = cfg.bs_train
        b = cfg.bs_load
    elif 'val' in ds_type:
        size = cfg.bs_val
        b = 1
    elif 'test' in ds_type:
        size = cfg.bs_test
        b = 1
    else:
        raise ValueError
    fn = dsFilename(
        fx_type='mfx',
        ds_type=ds_type,
        b=b,
        m=cfg.max_m,
        n=cfg.max_n,
        d=cfg.max_d,
        q=cfg.max_q,
        size=size,
        part=part,
        tag=cfg.d_tag,
        suffix=suffix,
        outside=True,
    )
    return fn


def sampleInt(batch_size: int | tuple, min_val: int, max_val: int) -> torch.Tensor:
    '''sample batch from discrete uniform
    (include the max_val as possible value)'''
    shape = (batch_size,) if isinstance(batch_size, int) else batch_size
    return torch.randint(min_val, max_val + 1, shape)


def sampleIntBatched(
    min_val: int | torch.Tensor, max_val: int | torch.Tensor
) -> torch.Tensor:
    '''batched version of the above with varying bounds'''
    if isinstance(min_val, int):
        min_val = torch.tensor([min_val])
    if isinstance(max_val, int):
        max_val = torch.tensor([max_val])
    n = max(len(min_val), len(max_val))
    samples = min_val + torch.floor((max_val + 1 - min_val) * torch.rand(n)).long()
    return samples


def sampleHN(shape: tuple, scale: float, max_val: float = float('inf')) -> torch.Tensor:
    '''sample batch from halfnormal, optionally clamp'''
    val = D.HalfNormal(scale).sample(shape)
    if max_val < float('inf'):
        val = val.clamp(max=max_val) # type: ignore
    return val # type: ignore


# -----------------------------------------------------------------------------
# aggregation

def maxShapes(data: list[dict[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    ''' get the maximal shapes over the whole list for per key '''
    max_shapes = {}
    for entry in data:
        for key, tensor in entry.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError('expected all entries to be tensors')
            shape = tuple(tensor.shape)
            if key not in max_shapes:
                max_shapes[key] = shape
            else:
                # Expand max_shapes[key] tuple elementwise to max dimension
                max_shapes[key] = tuple(
                    max(old_dim, new_dim)
                    for old_dim, new_dim in zip(max_shapes[key], shape)
                )
    return max_shapes


def aggregate(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    ''' finds out maximal dimensions for each key in list of tensor-dicts
        collates all tensor-dicts to single tensor-dicts with padded tensors '''
    max_shapes = maxShapes(data)
    batch_size = len(data)
    ds = {
        k: torch.empty(batch_size, *shape, dtype=data[0][k].dtype)
        for k, shape in max_shapes.items()
    }
    for i, item in enumerate(data):
        for k in ds.keys():
            shape = max_shapes[k]
            ds[k][i] = padTensor(item[k], shape)
    return ds


# -----------------------------------------------------------------------------
# wrapper

def generate(
    batch_size: int,
    seed: int, # seed for dataset simulation, separate from pymc seed
    fit: bool = False,
    bs_load: int = 1,
) -> list[dict[str, torch.Tensor]]:
    '''Generate a [batch_size] list of mixed effects datasets
    with varying n, m, d, q.'''

    # init
    torch.manual_seed(seed)
    data = []
    iterator = tqdm(range(batch_size))
    if not fit:
        iterator.set_description(f'{part:02d}/{cfg.iterations:02d}')

    # presample sizes
    d = sampleInt(batch_size, cfg.max_d, cfg.max_d)
    q = sampleInt(batch_size, cfg.max_q, cfg.max_q)
    m = sampleInt(batch_size // bs_load, cfg.min_m, cfg.max_m).repeat_interleave(
        bs_load
    )
    n = sampleInt((batch_size, cfg.max_m), cfg.min_n, cfg.max_n)
    d, m, n, q = d.tolist(), m.tolist(), n.tolist(), q.tolist()
 
    # presample normalized priors
    nu_ffx = D.Uniform(-3, 3).sample((batch_size, cfg.max_d))
    tau_ffx = D.Uniform(0.01, 3).sample((batch_size, cfg.max_d))
    tau_rfx = D.Uniform(0.01, 3).sample((batch_size, cfg.max_d))
    tau_eps = D.Uniform(0.01, 2).sample((batch_size,))

    # sample datasets
    for i in iterator:
        ds = None
        attempts = 0
        okay = False

        # setup parameter sampler
        d_i, q_i, n_i = d[i], q[i], n[i][: m[i]]
        prior = Prior(nu_ffx[i, :d_i],
                      tau_ffx[i, :d_i],
                      tau_eps[i],
                      tau_rfx[i, :q_i])

        # setup design matrix sampler
        if (cfg.semi or cfg.sub) and (seed <= 0 or torch.rand(1) < 0.5):
            design = Emulator(source=cfg.d_tag, use_sgld=cfg.sgld)
        else:
            design = Synthesizer(toy=cfg.toy)
 
        # generation loop
        while not okay:
            attempts += 1
            ds = Generator(prior, design, n_i, sub=cfg.sub, tag=i).sample()
            okay = ds['okay'] or attempts >= 20
        data += [ds]

    # optionally fit mcmc
    if fit:
        fitter = {'pymc': fitPyMC, 'bambi': fitBambi}[cfg.api]
        
        # HMC/NUTS
        print(f'Fitting NUTS using {cfg.api.upper()}')
        for ds in tqdm(data):
            nuts_results = fitter(ds, method='nuts', seed=cfg.seed,
                                  specify_priors=(not cfg.sub),
                                  use_multiprocessing=(not cfg.slurm))
            ds.update(nuts_results)
        
        # ADVI
        print(f'Fitting ADVI using {cfg.api.upper()}')
        for ds in tqdm(data):
            advi_results = fitter(ds, method='advi', seed=cfg.seed, 
                                  specify_priors=(not cfg.sub))
            ds.update(advi_results)
    return data


# =============================================================================
if __name__ == '__main__':
    
    
    # init outout directory
    Path('..', 'outputs', 'data').mkdir(parents=True, exist_ok=True)

    # setup config
    cfg = setup()
    assert cfg.bs_train % cfg.bs_load == 0, (
        'storage batch size must be divisible by loading batch size'
    )
    if cfg.semi:
        assert not cfg.toy, 'toy dataset is purely synthetic'
        assert not cfg.sub, 'sub-sampled dataset has no simulated parameters'
        assert cfg.d_tag, 'must specify dataset name in d_tag flag'
    elif cfg.sub:
        assert cfg.d_tag, 'must specify dataset name in d_tag flag'
        assert cfg.begin == -1, 'sub-sampled real data is only valid for test set'
        
    part = cfg.begin
    raw = None

    # generate test dataset
    if cfg.begin == -1:
        print('Generating test set...')
        ds_test = generate(cfg.bs_test, part, fit=True)
        ds_test = aggregate(ds_test)
        fn = getFileName(f'test{"-sub" if cfg.sub else ""}', -1)
        np.savez_compressed(fn, **ds_test, allow_pickle=True)
        print(f'\nSaved test set to {fn}')
        exit()

    # generate validation dataset
    if cfg.begin == 0:
        print('Generating validation set...')
        ds_val = generate(cfg.bs_val, part)
        ds_val = aggregate(ds_val)
        fn = getFileName('val', 0)
        np.savez_compressed(fn, **ds_val, allow_pickle=True)
        print(f'\nSaved validation set to {fn}')
        cfg.begin += 1

    # potentially stop after that
    if cfg.iterations == 0:
        exit()

    # generate training datasets
    print(f'Generating {cfg.iterations} training partitions of {cfg.bs_train} datasets each...')
    for part in range(cfg.begin, cfg.iterations + 1):
        ds_train = generate(cfg.bs_train, part, bs_load=cfg.bs_load)
        ds_train = aggregate(ds_train)
        fn = getFileName('train', part)
        np.savez_compressed(fn, **ds_train, allow_pickle=True)
        print(f'Saved training set to {fn}')

