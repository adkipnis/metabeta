from __future__ import annotations

import numpy as np

from metabeta.utils.evaluation import simulatorClipped
from metabeta.utils.families import POISSON_ETA_CLIP_MAX


def _write(path, lf: int, with_params: bool = True) -> None:
    # three datasets, two groups, intercept + one slope with a random intercept (q = 1):
    # η = 1 + x·β1 + α_g; dataset 1 crosses the clip at its last observation, dataset 2 only
    # in a padded observation beyond n
    X = np.zeros((3, 4, 2), dtype=np.float32)
    X[..., 0] = 1.0
    X[..., 1] = [[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 0.0, 50.0]]
    groups = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=np.int32)
    ffx = np.array([[1.0, 1.0], [1.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    rfx = np.array([[[0.0], [0.5]], [[0.0], [2.0]], [[0.0], [0.5]]], dtype=np.float32)
    raw = {
        'X': X,
        'groups': groups,
        'n': np.array([3, 3, 3], dtype=np.int32),
        'y': np.zeros((3, 4), dtype=np.float32),
        'likelihood_family': np.full(3, lf, dtype=np.int32),
    }
    if with_params:
        raw |= {'ffx': ffx, 'rfx': rfx}
    np.savez(path, **raw)


def test_flags_valid_observations_above_the_clip(tmp_path):
    path = tmp_path / 'test.npz'
    _write(path, lf=2)
    # dataset 1: η = 1 + 2·4 + 2 = 11 > 10; dataset 2: η = 51 only at the padded observation
    assert POISSON_ETA_CLIP_MAX == 10.0
    np.testing.assert_array_equal(simulatorClipped(path), [False, True, False])


def test_other_families_and_real_data_are_never_clipped(tmp_path):
    for lf, with_params in ((0, True), (1, True), (2, False)):
        path = tmp_path / f'test_{lf}_{with_params}.npz'
        _write(path, lf=lf, with_params=with_params)
        np.testing.assert_array_equal(simulatorClipped(path), [False, False, False])
