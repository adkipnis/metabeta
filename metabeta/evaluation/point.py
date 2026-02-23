from typing import cast
import torch
from scipy.stats import pearsonr
import numpy as np

from metabeta.utils.evaluation import Proposal, getMasks, weightedQuantile


def maskedMean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(x.dim() - 1))
    n = mask.sum(dims).clamp_min(1.0)
    return (x * mask).sum(dims) / n


def maskedStd(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(x.dim() - 1))
    n = mask.sum(dims).clamp_min(1.0)
    mean = (x * mask).sum(dims) / n
    square_diff = (x - mean).square() * mask
    var = square_diff.sum(dims) / n
    return torch.sqrt(var)


def pointEstimate(
    x: torch.Tensor, w: torch.Tensor | None = None, method: str = 'mean'
) -> torch.Tensor:
    if method == 'mean':
        if w is None:
            return torch.mean(x, dim=-2)
        if x.dim() == 4:   # handle groups in rfx
            w = w.unsqueeze(1)
        return (x * w.unsqueeze(-1)).sum(-2)
    elif method == 'median':
        if w is None:
            return torch.median(x, dim=-2)[0]
        else:
            return weightedQuantile(x, w, q=0.5)
    else:
        raise NotImplementedError(f'method {method} not implemented')


def getMAP(x: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    """sample-based maximum a-posteriori estimate"""
    idx = log_prob.argmax(-1, keepdim=True)
    shape = (*idx.shape, x.shape[-1])
    idx_ext = idx.unsqueeze(-1).expand(shape)
    return x.gather(dim=-2, index=idx_ext).squeeze(-2)


def getPointEstimates(proposal: Proposal, method: str) -> dict[str, torch.Tensor]:
    if method != 'map':
        w = proposal.weights
        global_est = pointEstimate(proposal.samples_g, w, method)
        out = proposal.partition(global_est)
        out['rfx'] = pointEstimate(proposal.samples_l, w, method)
    else:
        global_est = getMAP(proposal.samples_g, proposal.log_prob_g)
        out = proposal.partition(global_est)
        out['rfx'] = getMAP(proposal.samples_l, proposal.log_prob_l)
    return out


def getRMSE(
    ests: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    out = {}
    masks = getMasks(data)
    for key, est in ests.items():
        gt = data[key]   # ground truth
        mask = masks[key]
        se = (gt - est).square()
        if mask is not None:
            mse = maskedMean(se, mask)
            norm = maskedStd(gt, mask)
        else:
            mse = se.mean(0)
            norm = gt.std(0, unbiased=False)
        rmse = torch.sqrt(mse)
        if normalize:
            rmse /= norm
        out[key] = rmse
    return out


def getCorrelation(
    locs: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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
        out[key] = torch.tensor(corr)
    return out
