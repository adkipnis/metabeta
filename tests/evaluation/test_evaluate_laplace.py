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


def test_mb_summary_cache_path_includes_run_options_and_mask(tmp_path):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(
        n_samples=1000,
        seed=7,
        k=0,
        pred_coverage=True,
    )
    evaluator.run_name = 'data=small-n-mixed_model=large_seed=13'
    evaluator.checkpoint_prefix = 'best'
    evaluator.data_path_test = tmp_path / 'small-n-sampled' / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    path_all = evaluator._summaryCachePath('test', 'mb', mask=None)
    path_masked = evaluator._summaryCachePath(
        'test',
        'mb',
        mask=np.array([True, False, True]),
    )

    assert path_all.parent == evaluator.data_path_test.parent
    assert 'summary_test_mb_data=small-n-mixed_model=large_seed=13_best' in path_all.name
    assert '_s1000_seed7_k0_predcov1_all.pt' in path_all.name
    assert path_masked != path_all
    assert path_masked.name.endswith('.pt')


def test_mb_summary_cache_candidates_include_legacy_checkpoint_name(tmp_path):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(
        n_samples=1000,
        seed=0,
        k=0,
        pred_coverage=True,
    )
    evaluator.run_name = 'data=small-n-mixed_model=large_seed=13'
    evaluator.legacy_run_name = 'data=small-n-mixed_model=large_seed=0'
    evaluator.checkpoint_prefix = 'best'
    evaluator.data_path_test = tmp_path / 'small-n-sampled' / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    candidates = evaluator._summaryCacheCandidates(
        'test',
        'mb',
        mask=np.array([True, True, True]),
    )

    names = [path.name for path in candidates]
    assert any('large_seed=13_best' in name for name in names)
    assert any('large_seed=0_best' in name for name in names)
    assert any('large_seed=0_latest' in name for name in names)
