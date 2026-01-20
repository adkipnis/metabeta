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

def datasetFilename(args: Namespace, epoch: int = 0) -> str:
 
    # partition type and optionally epoch
    parts = [args.partition]
    if args.partition == 'train':
        parts.append(f'ep{epoch:04d}')

    # sizes
    parts += [
        f'd{args.max_d}',
        f'q{args.max_q}',
        f'm{args.min_m}-{args.max_m}',
        f'n{args.min_n}-{args.max_n}',
        args.type,
    ]

    # source of sampled data
    if args.type == 'sampled':
        parts.append(args.source)

    return '_'.join(parts) + '.npz'
