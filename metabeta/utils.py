from pathlib import Path
import numpy as np
import torch

# --- setters and getters
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

# --- name utils
def dataname(
    d: int, # max number of fixed effects
    q: int, # max number of random effects
    m: int, # max number of groups
    n: int, # max number of observations per group
    b: int, # batch size
    p: int = 0, # partition number for trainin set
    fx_type: str = 'mfx',
    p_type: str = 'train', # [train, val, test]
    tag: str = '',
) -> str:
    if tag:
        tag += '-'
    if p:
        assert p_type == 'train'
        part = f'-part={p}'
    else:
        part = ''
    return f'{fx_type}-{p_type}-{tag}d={d}-q={q}-m={m}-n={n}-b={b}{part}.npz'

# --- sampling
def logUniform(a: float,
               b: float,
               add: float = 0.0,
               round: bool = False) -> float|int:
    assert a > 0, 'lower bound must be positive'
    assert b > a, 'upper bound must be larger than lower bound'
    out = np.exp(np.random.uniform(np.log(a), np.log(b)))
    out += add
    if round:
        return int(np.round(out))
    return float(out)


