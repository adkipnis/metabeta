import numpy as np
import torch

from metabeta.evaluation.point import getCorrelation
from metabeta.utils.regularization import corrToLower


def test_get_correlation_corr_rfx_uses_active_q_count_not_fixed_slot() -> None:
    q_max = 5
    q_i = 3
    qperm = np.array([0, 3, 1, 4, 2])
    mask_q = torch.as_tensor(
        np.repeat(((np.arange(q_max) < q_i)[qperm])[None, :], 2, axis=0), dtype=torch.bool
    )

    corr_true = []
    corr_est = []
    for rho, est_rho in ((0.1, 0.15), (0.4, 0.35)):
        corr = np.eye(q_max, dtype=np.float32)
        corr[1, 0] = corr[0, 1] = rho
        corr[2, 0] = corr[0, 2] = rho + 0.1
        corr[2, 1] = corr[1, 2] = rho + 0.2
        corr = corr[np.ix_(qperm, qperm)]
        corr_true.append(corr)

        est = np.eye(q_max, dtype=np.float32)
        est[1, 0] = est[0, 1] = est_rho
        est[2, 0] = est[0, 2] = est_rho + 0.1
        est[2, 1] = est[1, 2] = est_rho + 0.2
        est = est[np.ix_(qperm, qperm)]
        corr_est.append(est)

    data = {
        'mask_d': torch.ones((2, 1), dtype=torch.bool),
        'mask_q': mask_q,
        'mask_m': torch.ones((2, 1), dtype=torch.bool),
        'mask_mq': mask_q.unsqueeze(-2),
        'corr_rfx': torch.as_tensor(np.stack(corr_true), dtype=torch.float32),
    }
    locs = {'corr_rfx': corrToLower(torch.as_tensor(np.stack(corr_est), dtype=torch.float32))}

    out = getCorrelation(locs, data)
    active_pair_idx = [1, 6, 8]

    assert out['corr_rfx'].shape == (q_max * (q_max - 1) // 2,)
    assert torch.isfinite(out['corr_rfx'][active_pair_idx]).all()
    assert torch.allclose(out['corr_rfx'][active_pair_idx], torch.ones(len(active_pair_idx)))
