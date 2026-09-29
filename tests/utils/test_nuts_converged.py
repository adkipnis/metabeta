from __future__ import annotations

import numpy as np
import torch

from metabeta.utils.evaluation import nutsConvergeMask, nutsConverged


def _diag(prefix: str = 'nuts', **overrides) -> dict[str, np.ndarray]:
    # four datasets: clean, high R-hat, low tail ESS (padded entry ignored), one divergence
    base = {
        'rhat': np.array([[1.0, 1.005], [1.0, 1.02], [1.0, 1.0], [1.0, 1.0]]),
        'ess': np.full((4, 2), 900.0),
        'ess_tail': np.array([[900.0, 900.0], [900.0, 900.0], [900.0, 350.0], [900.0, 0.0]]),
        'divergences': np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
        'draws': np.full(4, 1000),
        'level': np.zeros(4, dtype=int),
    }
    base.update(overrides)
    return {f'{prefix}_{k}': v for k, v in base.items()}


def test_single_criterion_flags_each_failure():
    np.testing.assert_array_equal(nutsConverged(_diag()), [True, False, False, False])


def test_level_two_tolerates_a_divergence_rate_of_one_permille():
    diag = _diag(level=np.full(4, 2), draws=np.full(4, 2000))  # 8000 draws, 1 divergence
    np.testing.assert_array_equal(nutsConverged(diag), [True, False, False, True])
    diag = _diag(level=np.full(4, 1), draws=np.full(4, 2000))
    assert not nutsConverged(diag)[3]


def test_prefix_selects_the_level_file_keys():
    diag = _diag(prefix='nuts2', level=np.full(4, 2))  # 1/4000 divergences pass at level 2
    assert nutsConverged(diag, prefix='nuts2').tolist() == [True, False, False, True]


def test_mask_from_collated_batch_matches_numpy_core():
    diag = _diag()
    batch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in diag.items()}
    np.testing.assert_array_equal(nutsConvergeMask(batch), nutsConverged(diag))
