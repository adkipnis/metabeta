import torch
from torch.nn import functional as F
from scipy.stats import pearsonr
import numpy as np


Proposed = dict[str, dict[str, torch.Tensor]]


def numFixed(proposed: Proposed) -> int:
    q = proposed['local']['samples'].shape[-1]
    D = proposed['global']['samples'].shape[-1]
    d = D - q - 1
    return int(d)


def sampleLoc(
    proposed: Proposed, loc_type: str = 'mean'
) -> dict[str, torch.Tensor]:
    d = numFixed(proposed)
    loc_fn = {'mean': torch.mean, 'median': torch.median}[loc_type]
    samples_g = proposed['global']['samples']
    samples_l = proposed['local']['samples']

    rfx_loc = loc_fn(samples_l, dim=-2)
    global_loc = loc_fn(samples_g, dim=-2)
    ffx_loc = global_loc[..., :d]
    sigma_rfx_loc = global_loc[..., d:-1]
    sigma_eps_loc = global_loc[..., -1]

    return {
        'ffx': ffx_loc,
        'sigma_rfx': sigma_rfx_loc,
        'sigma_eps': sigma_eps_loc,
        'rfx': rfx_loc,
    }


# def sampleStd(proposed: Proposed) -> dict[str, torch.Tensor]:
#     d = numFixed(proposed)
#     global_std = proposed['global']['samples'].std(-2)
#     ffx_std = global_std[..., :d]
#     sigma_rfx_std = global_std[..., d:-1]
#     sigma_eps_std = global_std[..., -1]
#     rfx_std = proposed['local']['samples'].std(-2)
#     return {
#         'ffx': ffx_std,
#         'sigma_rfx': sigma_rfx_std,
#         'sigma_eps': sigma_eps_std,
#         'rfx': rfx_std,
#     }


def getMasks(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    out = {}
    out['ffx'] = data['mask_d']
    out['sigma_rfx'] = data['mask_q']
    out['sigma_eps'] = None
    out['rfx'] = data['mask_m'].unsqueeze(-1) * data['mask_q'].unsqueeze(-2)
    return out


def sampleRMSE(
    data: dict[str, torch.Tensor],
    locs: dict[str, torch.Tensor],
) -> dict[str, float]:
    out = {}
    masks = getMasks(data)
    for key, est in locs.items():
        gt = data[key] # ground truth
        mask = masks[key]
        if mask is not None:
            se = F.mse_loss(gt, est, reduction='none')
            rmse = torch.sqrt(se.sum() / mask.sum()).item()
        else:
            rmse = torch.sqrt(F.mse_loss(gt, est))
        out[key] = float(rmse)
    return out


def sampleCorrelation(
    data: dict[str, torch.Tensor],
    locs: dict[str, torch.Tensor],
) -> dict[str, float]:
    out = {}
    masks = getMasks(data)
    for key, est in locs.items():
        gt = data[key]
        mask = masks[key]
        if mask is not None:
            D = mask.shape[-1]
            corr = np.zeros(D)
            for i in range(D):
                mask_i = mask[..., i]
                gt_i = gt[mask_i, i]
                est_i = est[mask_i, i]
                corr[i] = pearsonr(gt_i, est_i)[0]
        else:
            corr = pearsonr(gt, est)[0]
        out[key] = float(corr.mean())
    return out
