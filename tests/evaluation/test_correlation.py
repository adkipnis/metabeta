import numpy as np
import torch

from metabeta.evaluation.correlation import (
    _corrSamplingDistribution,
    evaluateCorrelation,
)


def test_corr_sampling_distribution_tracks_target_rho():
    rs = _corrSamplingDistribution(m=60, rho=0.65, n_sim=4000, seed=0)
    assert abs(rs.mean() - 0.65) < 0.05


def test_evaluate_correlation_returns_bounds_and_percentiles():
    b, m, s, q = 2, 12, 128, 2
    rhos = [0.0, 0.7]
    rng = np.random.default_rng(0)

    rfx = np.zeros((b, m, s, q), dtype=np.float32)
    corr_true = np.zeros((b, q, q), dtype=np.float32)
    for i, rho in enumerate(rhos):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        corr_true[i] = cov
        for k in range(s):
            rfx[i, :, k, :] = rng.multivariate_normal(np.zeros(2), cov, size=m)

    rfx_t = torch.from_numpy(rfx)
    data = {
        'mask_m': torch.ones((b, m), dtype=torch.bool),
        'corr_rfx': torch.from_numpy(corr_true),
        'eta_rfx': torch.tensor([0.0, 1.3]),
        'm': torch.tensor([m, m]),
        'rfx': rfx_t[:, :, 0, :],  # (b, m, q) true rfx for oracle baseline
    }

    out = evaluateCorrelation(rfx_t, data, n_sim=1200)

    assert out['corr_true'].shape == (b, 1)
    assert out['corr_mean'].shape == (b, 1)
    assert out['corr_q025'].shape == (b, 1)
    assert out['corr_q975'].shape == (b, 1)
    assert out['percentile'].shape == (b,)
    assert out['percentile_pairs'].shape == (b, 1)

    inside = (out['corr_mean'] >= out['corr_q025']) & (out['corr_mean'] <= out['corr_q975'])
    assert inside.float().mean().item() > 0.5
