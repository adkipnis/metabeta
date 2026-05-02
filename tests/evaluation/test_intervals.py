import numpy as np
import torch

from metabeta.evaluation.intervals import getCoveragePerParameter
from metabeta.utils.regularization import corrToLower


def test_corr_rfx_coverage_respects_permuted_mask_q_for_q_ge_3() -> None:
    q_i = 3
    q_max = 5

    corr = np.eye(q_max, dtype=np.float32)
    for (i, j), value in {(1, 0): 0.1, (2, 0): 0.2, (2, 1): 0.3}.items():
        corr[i, j] = corr[j, i] = value

    qperm = np.array([0, 3, 1, 4, 2])
    corr_perm = corr[np.ix_(qperm, qperm)]
    mask_q = torch.as_tensor((np.arange(q_max) < q_i)[qperm], dtype=torch.bool).unsqueeze(0)
    corr_t = torch.as_tensor(corr_perm, dtype=torch.float32).unsqueeze(0)

    gt = corrToLower(corr_t)[0]
    ci = torch.full((1, 2, gt.numel()), 1.5, dtype=torch.float32)
    ci[:, 1] = 2.0

    active_pair_idx = [1, 6, 8]
    ci[0, 0, active_pair_idx] = gt[active_pair_idx] - 1e-3
    ci[0, 1, active_pair_idx] = gt[active_pair_idx] + 1e-3

    data = {
        'mask_d': torch.ones((1, 1), dtype=torch.bool),
        'mask_q': mask_q,
        'mask_m': torch.ones((1, 1), dtype=torch.bool),
        'mask_mq': mask_q.unsqueeze(-2),
        'corr_rfx': corr_t,
    }
    coverage = getCoveragePerParameter({'corr_rfx': ci}, data)

    expected = torch.zeros_like(gt)
    expected[active_pair_idx] = 1.0
    assert torch.equal(coverage['corr_rfx'], expected)
