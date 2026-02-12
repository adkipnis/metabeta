from typing import cast
import torch
from torch.nn import functional as F
from scipy.stats import pearsonr
import numpy as np

from metabeta.utils.evaluation import Proposal, getMasks


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
            return torch.median(x, dim=-2)[0]
        else:
            # TODO
            raise NotImplementedError 
    else:
        raise NotImplementedError(f'location type {loc_type} not implemented')


def sampleLoc(
    proposal: Proposal,
    loc_type: str = 'mean'
) -> dict[str, torch.Tensor]:
    samples_g = proposal.samples('global').clone()
    samples_l = proposal.samples('local')
    w = proposal.weights()

    # get location
    global_loc = getLocation(samples_g, w, loc_type)
    out = proposal.partition(global_loc)
    out['rfx'] = getLocation(samples_l, w, loc_type)
    return out


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

