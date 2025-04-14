from pathlib import Path
from typing import Tuple
import torch


def dInput(d_data: int, fx_type: str) -> int:
    n_fx = 2 if fx_type == 'mfx' else 1
    return 1 + n_fx * (1 + d_data)

def getAlpha(noise_std: torch.Tensor) -> torch.Tensor:
    b, n  = noise_std.shape
    ns = torch.stack([torch.arange(n)] * b)
    return 2. + ns / 2.


def moments2ig(loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' reparameterize IG-distribution (from loc and scale to alpha and beta) '''
    eps = torch.finfo(loc.dtype).eps
    alpha = 2. + torch.div(loc.square(), scale.square().clamp(min=eps))
    beta = torch.mul(loc, alpha - 1.)
    return alpha, beta


def causalMask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))


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


def dsFilename(fx_type: str, ds_type: str, d: int, n: int, size: int, part: int) -> Path:
    ''' example: ffx-train-d=8-n=50-10k-001.pt'''
    s = parseSize(size)
    p = f'{part:03d}'
    return Path('data', f'{fx_type}-{ds_type}-d={d}-n={n}-{s}-{p}.pt')


# def averageOverN(losses: torch.Tensor,
#                  n: int,
#                  b: int,
#                  depths: torch.Tensor,
#                  n_min: int = 0,
#                  weigh: bool = False) -> torch.Tensor:
#     ''' mask out first n_min losses per batch,
#     then calculate weighted average over n with higher emphasis on later n'''
#     # losses (b, n, d)
#     if cfg.tol == 0:
#         return losses.mean(dim=1)
#     if n_min == 0:
#         n_min = cfg.tol * depths.unsqueeze(1) # (b, 1)
#     denominators = n - n_min # (b, 1)
#     mask = torch.arange(n).expand(b, n) < n_min # (b, n)
#     losses[mask] = 0.
#     if weigh:
#         weights = torch.arange(0, 1, 1/n) + 1/n
#         weights = weights.sqrt().unsqueeze(0).unsqueeze(-1)
#         losses = losses * weights
#     losses = losses.sum(dim=1) / denominators
#     return losses # (batch, d)


# def assembleNoiseInputs(y: torch.Tensor, X: torch.Tensor,
#                         outputs: Dict[str, torch.Tensor],
#                         posterior_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
#     if posterior_type == "discrete":
#         logits = outputs["ffx_logits"]
#         loc = dprop_normal.mean(logits).detach()
#         scale = dprop_normal.variance(logits, loc).detach().sqrt()
#     elif posterior_type == "mixture":
#         locs = outputs["ffx_loc"]
#         scales = outputs["ffx_scale"]
#         weights = outputs["ffx_weight"]
#         loc = mprop.mean(locs, scales, weights).detach()
#         scale = mprop.variance(locs, scales, weights, loc).sqrt().detach()
#     else:
#         raise ValueError(f"posterior type {posterior_type} not supported.")
#     y_pred = torch.sum(X * loc, dim=-1).unsqueeze(-1)
#     res = y - y_pred
#     return res, scale