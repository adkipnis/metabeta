import pandas as pd
import torch
import bambi as bmb
from bambi import Prior
import arviz as az
import time

from torch import distributions as D
import matplotlib.pyplot as plt

def pandify(ds: dict[str, torch.Tensor]) -> pd.DataFrame:
    n, d = int(ds['n']), int(ds['d'])
    df = pd.DataFrame(index=range(n))
    df['i'] = ds['groups'].numpy()
    df['y'] = ds['y'].unsqueeze(-1).numpy()
    for j in range(1, d):
        df[f'x{j}'] = ds['X'][..., j].numpy()
    return df


def formulate(ds: dict[str, torch.Tensor]) -> str:
    d, q = int(ds['d']), int(ds['q'])
    fixed = ' + '.join(f'x{j}' for j in range(1, d))
    random = ' + '.join(f'x{j}' for j in range(1, q))
    if not random:
        random = '1'
    out = f'y ~ 1 + {fixed} + ({random} | i)'
    return out


