import argparse
import numpy as np
import pytest
import torch

from metabeta.evaluation.evaluate import Evaluator
from metabeta.utils.evaluation import AggregatedMetrics, EvaluationSummary, PerDatasetMetrics


def _summary(tpd=None):
    return EvaluationSummary(
        per_dataset=PerDatasetMetrics(
            posterior_nll=torch.tensor([1.0, 2.0, 3.0]),
            loo_nll=torch.tensor([1.5, 2.5, 3.5]),
            pp_fit=torch.tensor([0.1, 0.2, 0.3]),
        ),
        aggregated=AggregatedMetrics(
            corr={'ffx': torch.tensor([0.5])},
            nrmse={'ffx': torch.tensor([0.6])},
            coverage={},
            ece={'ffx': torch.tensor([0.0])},
            eace={'ffx': torch.tensor([0.1])},
            lcr={},
            abs_lcr={},
            estimates={},
        ),
        tpd=tpd,
    )


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


@pytest.mark.parametrize(
    ('src_mask', 'common_mask', 'expected'),
    [
        (None, None, None),
        (np.array([True, True, True]), None, None),
        (np.array([True, False, True]), np.array([True, False, True]), None),
        (None, np.array([True, False, True]), np.array([True, False, True])),
        (
            np.array([True, True, False]),
            np.array([True, False, False]),
            np.array([True, False, False]),
        ),
    ],
)
def test_fit_summary_mask_uses_native_cache_when_possible(src_mask, common_mask, expected):
    result = Evaluator._fitSummaryMask(src_mask, common_mask, n=3)

    if expected is None:
        assert result is None
    else:
        np.testing.assert_array_equal(result, expected)


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


def test_make_row_includes_loo_nll_and_predictive_width():
    evaluator = Evaluator.__new__(Evaluator)
    summary = EvaluationSummary(
        per_dataset=PerDatasetMetrics(
            posterior_nll=torch.tensor([1.0, 3.0]),
            loo_nll=torch.tensor([2.0, 4.0]),
            pp_fit=torch.tensor([0.1, 0.2]),
            pp_cov_width=torch.tensor([[9.0, 11.0], [5.0, 7.0]]),
        ),
        aggregated=AggregatedMetrics(
            corr={'ffx': torch.tensor([0.5])},
            nrmse={'ffx': torch.tensor([0.6])},
            coverage={},
            ece={'ffx': torch.tensor([0.0])},
            eace={'ffx': torch.tensor([0.1])},
            lcr={},
            abs_lcr={},
            estimates={},
        ),
        tpd=0.25,
    )

    row = evaluator._makeRow('LAPLACE', summary, 'ppR2')

    assert row['LOO-NLL'] == pytest.approx(2.0)
    assert row['ppNLL'] == pytest.approx(1.0)
    assert row['ppWidth90'] == pytest.approx(5.0)
    assert row['tpd'] == pytest.approx(0.25)
    assert row['EACE'] == pytest.approx(0.1)


def test_save_tables_accepts_loo_nll(tmp_path):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(likelihood_family=0)
    evaluator.results_dir = tmp_path

    rows = [
        {
            'method': 'MB',
            'R': 0.9,
            'NRMSE': 0.2,
            'ECE': 0.01,
            'EACE': 0.02,
            'RFX_joint_ECE': 0.01,
            'RFX_joint_EACE': 0.02,
            'LOO-NLL': 1.5,
            'ppNLL': 1.4,
            'ppR2': 0.3,
            'tpd': 0.1,
            'IS_eff': None,
            'Pareto_k': None,
            'ppEACE': 0.02,
            'ppWidth90': 4.0,
        },
        {
            'method': 'LAPLACE',
            'R': 0.8,
            'NRMSE': 0.3,
            'ECE': 0.02,
            'EACE': 0.01,
            'RFX_joint_ECE': 0.02,
            'RFX_joint_EACE': 0.03,
            'LOO-NLL': 1.3,
            'ppNLL': 1.2,
            'ppR2': 0.2,
            'tpd': 0.01,
            'IS_eff': None,
            'Pareto_k': None,
            'ppEACE': 0.03,
            'ppWidth90': 3.0,
        },
    ]

    evaluator.saveTables(rows)

    table = (tmp_path / 'evaluate.md').read_text()
    assert 'EACE' in table
    assert 'LOO-NLL' in table
    assert '**1.3000**' in table
    assert '**3.0000**' in table


def test_cached_rows_do_not_require_dataloader(tmp_path):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(
        n_samples=1000,
        seed=0,
        k=0,
        pred_coverage=False,
        likelihood_family=0,
    )
    evaluator.run_name = 'run'
    evaluator.legacy_run_name = 'run'
    evaluator.checkpoint_prefix = 'best'
    evaluator.ckpt_dir = None
    evaluator.dl_test = None
    evaluator.dl_valid = None
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    np.savez(
        evaluator.data_path_test,
        y=np.zeros((3, 4), dtype=np.float32),
        laplace_failed=np.array([False, False, False]),
        laplace_duration=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    _summary(tpd=1.0).save(evaluator._summaryCachePath('test', 'mb'))
    _summary(tpd=2.0).save(evaluator._summaryCachePath('test', 'nuts'))
    _summary(tpd=None).save(evaluator._summaryCachePath('test', 'laplace'))

    rows = evaluator._cachedRowsForPartition(
        'test',
        ['MB', 'NUTS', 'LAPLACE'],
        fit_label='ppR2',
        multi=False,
    )

    assert rows is not None
    assert [row['method'] for row in rows] == ['MB', 'NUTS', 'LAPLACE']
    assert rows[2]['tpd'] == pytest.approx(0.2)
    assert evaluator.dl_test is None


def test_mb_warmup_uses_one_sample_and_restores_torch_rng():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(warmup=True, k=0, n_samples=100, rescale=False)
    evaluator.device = torch.device('cpu')
    calls = []

    def sample_batch(batch, n_samples=None):
        calls.append(n_samples)
        torch.rand(1)
        return object()

    evaluator._sampleBatch = sample_batch
    batch = {'X': torch.zeros(2, 3)}

    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    evaluator._warmupMbBatch(batch, 'test')
    actual = torch.rand(3)

    assert calls == [1]
    torch.testing.assert_close(actual, expected)


def test_table_only_cache_miss_raises_before_full_evaluation(tmp_path):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(
        save_tables=True,
        plot=False,
        converged_subset=False,
        partition='test',
        models='MB,NUTS',
        likelihood_family=0,
    )
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = tmp_path / 'valid.fit.npz'
    evaluator._cachedRowsForPartition = lambda *args, **kwargs: None

    def fail_eval(*args, **kwargs):
        raise AssertionError('full evaluation should not be reached in table-only mode')

    evaluator._evalPartition = fail_eval

    with pytest.raises(RuntimeError, match='Refusing to fall back to full evaluation'):
        evaluator.go()


def test_table_only_cache_miss_can_fallback_for_mb_only():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(
        save_tables=True,
        plot=False,
        converged_subset=False,
        partition='test',
        models='MB',
        likelihood_family=0,
    )
    evaluator.data_path_test = 'test.fit.npz'
    evaluator.data_path_valid = 'valid.fit.npz'
    evaluator.results_dir = None
    evaluator._cachedRowsForPartition = lambda *args, **kwargs: None
    evaluator._hasFits = lambda partition: True
    evaluator._evalPartition = lambda *args, **kwargs: [{'method': 'MB'}]

    evaluator.go()


def test_mb_only_partition_data_uses_base_file(monkeypatch):
    class DummyLoader:
        def fullBatch(self):
            return {'X': torch.zeros(1, 2)}

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(batch_size=16)
    evaluator.dl_test = None
    evaluator.dl_valid = None
    calls = []

    def fake_loader(partition, batch_size=None, prefer_fit=True):
        calls.append((partition, batch_size, prefer_fit))
        suffix = 'fit.npz' if prefer_fit else 'npz'
        return DummyLoader(), f'{partition}.{suffix}'

    monkeypatch.setattr(evaluator, '_getDataLoader', fake_loader)

    _, _, path = evaluator._getPartitionData('test', need_fits=False)

    assert path == 'test.npz'
    assert calls == [('test', 16, False)]
