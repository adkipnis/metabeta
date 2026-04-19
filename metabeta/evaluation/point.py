from typing import cast
import torch
from scipy.stats import pearsonr
import numpy as np

from metabeta.utils.evaluation import Proposal, getMasks, weightedQuantile
from metabeta.utils.regularization import corrToLower


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


def getPointEstimates(proposal: Proposal, method: str) -> dict[str, torch.Tensor]:
    if method not in ('mean', 'median', 'hybrid'):
        raise ValueError(f"method must be 'mean', 'median', or 'hybrid', got '{method}'")
    w = proposal.weights
    if method == 'hybrid':
        # median for location params (ffx), mean for scale params (sigma_*) and rfx
        global_mean   = pointEstimate(proposal.samples_g, w, 'mean')
        global_median = pointEstimate(proposal.samples_g, w, 'median')
        out = proposal.partition(global_mean)
        out['ffx'] = proposal.partition(global_median)['ffx']
        out['rfx'] = pointEstimate(proposal.samples_l, w, 'mean')
    else:
        global_est = pointEstimate(proposal.samples_g, w, method)
        out = proposal.partition(global_est)
        out['rfx'] = pointEstimate(proposal.samples_l, w, method)

    if proposal.corr_rfx is not None:
        corr_samples = proposal.corr_rfx  # (b, n_s, q, q)
        w = proposal.weights
        if w is None:
            corr_mean = corr_samples.mean(dim=-3)
        else:
            corr_mean = (corr_samples * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=-3)
        out['corr_rfx'] = corrToLower(corr_mean)

    return out


def getRMSE(
    ests: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    out = {}
    masks = getMasks(data)
    for key, est in ests.items():
        if key == 'corr_rfx':
            gt = corrToLower(data['corr_rfx'])
            mask = None
        else:
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
        if key == 'corr_rfx':
            gt = corrToLower(data['corr_rfx'])  # (b, d_corr)
            corr = np.array([
                cast(np.float32, pearsonr(gt[:, k].numpy(), est[:, k].numpy())[0])
                for k in range(gt.shape[-1])
            ], dtype=np.float32)
            out[key] = torch.tensor(corr)
            continue
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
