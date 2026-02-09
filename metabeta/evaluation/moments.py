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

def sampleMean(proposed: Proposed) -> dict[str, torch.Tensor]:
    d = numFixed(proposed)
    global_mean = proposed['global']['samples'].mean(-2)
    ffx_mean = global_mean[..., :d]
    sigma_rfx_mean = global_mean[..., d:-1]
    sigma_eps_mean = global_mean[..., -1]
    rfx_mean = proposed['local']['samples'].mean(-2)
    return {
        'ffx': ffx_mean,
        'sigma_rfx': sigma_rfx_mean,
        'sigma_eps': sigma_eps_mean,
        'rfx': rfx_mean,
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
        means: dict[str, torch.Tensor],
        ) -> dict[str, float]:
    out = {}
    masks = getMasks(data)
    for key, estimated in means.items():
        ground_truth = data[key]
        mask = masks[key]
        if mask is not None:
            squared_error = F.mse_loss(ground_truth, estimated, reduction='none')
            rmse = torch.sqrt(squared_error.sum() / mask.sum()).item()
        else:
            rmse = torch.sqrt(F.mse_loss(ground_truth, estimated))
        out[key] = float(rmse)
    return out

