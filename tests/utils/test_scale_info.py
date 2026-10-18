import numpy as np
import torch

from metabeta.utils.api import ScaleInfo


def test_sd_map_without_correlations_stays_on_the_sds_device():
    # the meta device stands in for CUDA: an operand on the CPU raises there too
    si = ScaleInfo(
        y_mean=0.0,
        y_std=1.0,
        x_means=np.array([0.0, 1.0]),
        x_stds=np.array([1.0, 2.0]),
        param_names=['Intercept', 'x'],
        random_names=['Intercept'],
    )
    sigma, corr = si.sdCorrToOriginal(torch.ones(4, 1, device='meta'), None)
    assert sigma.device.type == 'meta' and corr is None
