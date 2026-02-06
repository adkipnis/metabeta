from argparse import Namespace
import numpy as np
import torch

def getDevice() -> str:
    if torch.cuda.is_available():
        return 'gpu'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def setSeed(s: int) -> np.random.Generator:
    torch.manual_seed(s)
    np.random.seed(s)
    rng = np.random.default_rng(s)
    return rng

def datasetFilename(args: Namespace, partition: str,  epoch: int = 0) -> str:
 
    # partition type and optionally epoch
    parts = [partition]
    if partition == 'train':
        parts.append(f'ep{epoch:04d}')

    # sizes
    parts += [
        f'd{args.max_d}',
        f'q{args.max_q}',
        f'm{args.min_m}-{args.max_m}',
        f'n{args.min_n}-{args.max_n}',
        args.ds_type,
    ]

    # source of sampled data
    if args.ds_type == 'sampled':
        parts.append(args.source)

    return '_'.join(parts) + '.npz'
