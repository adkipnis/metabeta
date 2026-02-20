from typing import cast
import torch
from torch.nn import functional as F
from scipy.stats import pearsonr
import numpy as np

from metabeta.utils.evaluation import Proposal, getMasks, weightedQuantile


def maskedMean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / n


def maskedStd(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum() / n
    square_diff = (x - mean).square() * mask
    var = square_diff.sum() / n
    return torch.sqrt(var)


def pointEstimate(
    x: torch.Tensor, w: torch.Tensor | None = None, loc_type: str = 'mean') -> torch.Tensor:
    if loc_type == 'mean':
        if w is None:
            return torch.mean(x, dim=-2)
        if x.dim() == 4:   # handle groups in rfx
            w = w.unsqueeze(1)
        return (x * w.unsqueeze(-1)).sum(-2)
    elif loc_type == 'median':
        if w is None:
            return torch.median(x, dim=-2)[0]
        else:
            return weightedQuantile(x, w, q=0.5)
    else:
        raise NotImplementedError(f'location type {loc_type} not implemented')


def getPointEstimates(proposal: Proposal, loc_type: str) -> dict[str, torch.Tensor]:
    w = proposal.weights
    global_loc = pointEstimate(proposal.samples_g, w, loc_type)
    out = proposal.partition(global_loc)
    out['rfx'] = pointEstimate(proposal.samples_l, w, loc_type)
    return out


def getRMSE(
    locs: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    normalize: bool = True,
    to_float: bool = True,
) -> dict[str, float | torch.Tensor]:
    out = {}
    masks = getMasks(data)
    for key, est in locs.items():
        gt = data[key]   # ground truth
        mask = masks[key]
        if mask is not None:
            se = (gt - est).square()
            mse = maskedMean(se, mask)
            rmse = torch.sqrt(mse)
            if normalize:
                rmse = rmse / maskedStd(gt, mask)
        else:
            rmse = torch.sqrt(F.mse_loss(gt, est))
            if normalize:
                rmse = rmse / gt.std(unbiased=False)
        if to_float:
            rmse = rmse.item()
        out[key] = rmse
    return out


def getCorrelation(
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
