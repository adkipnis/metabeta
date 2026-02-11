from typing import cast
import torch
from torch.nn import functional as F
from scipy.stats import pearsonr
import numpy as np

from metabeta.utils.evaluation import Proposed, numFixed, getMasks


def getLocation(
    x: torch.Tensor, w: torch.Tensor | None = None, loc_type: str = 'mean',
) -> torch.Tensor:
    if loc_type == 'mean':
        if w is None:
            return torch.mean(x, dim=-2)
        if x.dim() == 4: # handle groups in rfx
            w = w.unsqueeze(1)
        return (x * w.unsqueeze(-1)).sum(-2)
    elif loc_type == 'median':
        if w is None:
            return torch.median(x, dim=-2)
        ... # TODO
    else:
        raise NotImplementedError(f'location type {loc_type} not implemented')

def sampleLoc(
    proposed: Proposed, loc_type: str = 'mean'
) -> dict[str, torch.Tensor]:
    d = numFixed(proposed)
    loc_fn = {
        'mean': lambda x: torch.mean(x, dim=-2),
        'median': lambda x: torch.median(x, dim=-2)[0],
    }[loc_type]
    samples_g = proposed['global']['samples']
    samples_l = proposed['local']['samples']
    rfx_loc = loc_fn(samples_l)
    global_loc = loc_fn(samples_g)
    ffx_loc = global_loc[..., :d]
    sigma_rfx_loc = global_loc[..., d:-1]
    sigma_eps_loc = global_loc[..., -1]

    return {
        'ffx': ffx_loc,
        'sigma_rfx': sigma_rfx_loc,
        'sigma_eps': sigma_eps_loc,
        'rfx': rfx_loc,
    }


def sampleStd(proposed: Proposed) -> dict[str, torch.Tensor]:
    d = numFixed(proposed)
    global_std = proposed['global']['samples'].std(-2)
    ffx_std = global_std[..., :d]
    sigma_rfx_std = global_std[..., d:-1]
    sigma_eps_std = global_std[..., -1]
    rfx_std = proposed['local']['samples'].std(-2)
    return {
        'ffx': ffx_std,
        'sigma_rfx': sigma_rfx_std,
        'sigma_eps': sigma_eps_std,
        'rfx': rfx_std,
    }


def sampleRMSE(
    locs: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
) -> dict[str, float]:
    out = {}
    masks = getMasks(data)
    for key, est in locs.items():
        gt = data[key]   # ground truth
        mask = masks[key]
        if mask is not None:
            se = (gt - est).pow(2)
            se = se * mask
            n =  mask.sum().clamp_min(1.0)
            rmse = torch.sqrt(se.sum() / n).item()
        else:
            rmse = torch.sqrt(F.mse_loss(gt, est))
        out[key] = float(rmse)
    return out


def sampleCorrelation(
    locs: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
) -> dict[str, float]:
    out = {}
    masks = getMasks(data)
    for key, est in locs.items():
        gt = data[key]
        mask = masks[key]
        if mask is not None:
            D = mask.shape[-1]
            corr = np.zeros(D, dtype=np.float32)
            for i in range(D):
                mask_i = mask[..., i]
                gt_i = gt[mask_i, i]
                est_i = est[mask_i, i]
                corr[i] = cast(np.float32, pearsonr(gt_i, est_i)[0])
        else:
            corr = cast(np.float32, pearsonr(gt, est)[0])
        out[key] = float(corr.mean())
    return out
