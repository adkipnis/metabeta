import argparse
import numpy as np
import pytest

from metabeta.evaluation.evaluate import Evaluator


def test_evaluator_resolves_laplace_fit_model():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(models='LAPLACE')

    assert evaluator._resolveModels() == ['LAPLACE']


def test_common_mask_intersects_failed_fit_masks():
    mask_a = np.array([True, False, True, True])
    mask_b = np.array([True, True, False, True])

    common = Evaluator._commonMask([None, mask_a, mask_b], n=4)

    np.testing.assert_array_equal(common, np.array([True, False, False, True]))


@pytest.mark.parametrize(
    ('src_mask', 'comparison_mask', 'expected'),
    [
        (None, np.ones(3, dtype=bool), True),
        (np.array([True, False, True]), np.array([True, False, True]), True),
        (None, np.array([True, False, True]), False),
        (np.array([True, True, True]), np.array([True, False, True]), False),
    ],
)
def test_native_summary_cache_requires_same_mask(src_mask, comparison_mask, expected):
    native_mask = np.ones(3, dtype=bool) if src_mask is None else src_mask

    assert np.array_equal(native_mask, comparison_mask) is expected
