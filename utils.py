from pathlib import Path
import torch


def symmetricMatrix2Vector(matrix: torch.Tensor) -> torch.Tensor:
    r, c = matrix.shape
    indices = torch.tril_indices(row=r, col=c, offset=0)
    return matrix[indices[0], indices[1]]


def symmetricMatrixFromVector(vector: torch.Tensor, n: int) -> torch.Tensor:
    matrix = torch.zeros(n, n)
    indices = torch.tril_indices(row=n, col=n, offset=0)
    matrix[indices[0], indices[1]] = vector
    return matrix + matrix.t() - torch.diag(matrix.diag())


def parseSize(size: int) -> str:
    if size >= 1e6:
        n = f'{size/1e6:.0f}m'
    elif size >= 1e3:
        n = f'{size/1e3:.0f}k'
    else:
        n = str(size)
    return n


def parseVariance(fixed: float) -> str:
    if fixed == 0.:
        noise = "variable"
    else:
        noise = str(fixed)
    return noise


def dsFilename(ds_type: str, d: int, n: int, fixed: float, size: int, part: int) -> Path:
    ''' example: ffx-train-d=8-n=50-noise=fixed-10k-001.pt'''
    noise = parseVariance(fixed)
    s = parseSize(size)
    p = f'{part:03d}'
    return Path('data', f'{ds_type}-train-d={d}-n={n}-noise={noise}-{s}-{p}.pt')


def dsFilenameVal(ds_type: str, d: int, n: int, fixed: float) -> Path:
    noise = parseVariance(fixed)
    return Path('data', f'{ds_type}-val-d={d}-n={n}-noise={noise}.pt')


