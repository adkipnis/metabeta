import argparse

import numpy as np
import pytest
import torch

from metabeta.evaluation.evaluate import _ALL_MODELS, Evaluator
from metabeta.utils.results import Proposal


def _evaluator(**cfg_kwargs) -> Evaluator:
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(**cfg_kwargs)
    evaluator._imh_method_cache = None
    return evaluator


def _proposal(tpd: float | None = None) -> Proposal:
    proposal = Proposal(
        {
            'global': {'samples': torch.ones(2, 4, 2), 'log_prob': torch.zeros(2, 4)},
            'local': {'samples': torch.ones(2, 1, 4, 1), 'log_prob': torch.zeros(2, 1, 4)},
        },
        has_sigma_eps=False,
    )
    proposal.tpd = tpd
    return proposal


def test_resolve_models_accepts_mb_imh():
    evaluator = _evaluator(models='MB,MB+IMH,NUTS,ADVI')

    assert evaluator._resolveModels() == ['MB', 'MB+IMH', 'NUTS', 'ADVI']


def test_all_excludes_mb_imh():
    evaluator = _evaluator(models='all')

    assert evaluator._resolveModels() == list(_ALL_MODELS)
    assert 'MB+IMH' not in _ALL_MODELS


def test_resolve_models_rejects_unknown():
    evaluator = _evaluator(models='MB,IMH')

    with pytest.raises(ValueError, match='unknown model'):
        evaluator._resolveModels()


@pytest.mark.parametrize(
    ('lf', 'expected'), [(0, 'imhMarginal'), (1, 'imhLaplace'), (2, 'imhLaplace')]
)
def test_imh_method_defaults_come_from_presets(lf, expected):
    evaluator = _evaluator(likelihood_family=lf)

    assert evaluator._imhMethod() == expected


def test_imh_method_honours_explicit_override():
    evaluator = _evaluator(likelihood_family=0, imh_method='imhGlobal')

    assert evaluator._imhMethod() == 'imhGlobal'


def test_imh_method_rejects_family_mismatch():
    evaluator = _evaluator(likelihood_family=1, imh_method='imhGlobal')

    with pytest.raises(ValueError, match='incompatible with likelihood_family'):
        evaluator._imhMethod()


def test_method_name_maps_mb_imh_to_its_variant():
    evaluator = _evaluator(likelihood_family=0)

    assert evaluator._methodName('MB') == 'mb'
    assert evaluator._methodName('NUTS') == 'nuts'
    assert evaluator._methodName('MB+IMH') == 'imhMarginal'


def test_mb_imh_stays_active_without_fit_file(tmp_path):
    evaluator = _evaluator()
    evaluator.data_path_test = tmp_path / 'test.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    assert evaluator._activeModels('test', ['MB', 'MB+IMH', 'NUTS']) == ['MB', 'MB+IMH']


def test_imh_summary_cache_is_keyed_by_checkpoint(tmp_path):
    evaluator = _evaluator(n_samples=1000, seed=0, k=0, pred_coverage=False, likelihood_family=0)
    evaluator.run_name = 'data=small-n-mixed_model=large_seed=13'
    evaluator.checkpoint_prefix = 'latest'
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    imh_path = evaluator._summaryCachePath('test', 'imhMarginal', mask=None)
    nuts_path = evaluator._summaryCachePath('test', 'nuts', mask=None)

    assert 'summary_test_imhMarginal_data=small-n-mixed_model=large_seed=13_latest' in imh_path.name
    assert '_s1000_seed0_k0_predcov0_all.pt' in imh_path.name
    # fit methods stay keyed by the data file alone
    assert nuts_path.name == 'summary_test_nuts.pt'


def test_mb_summary_cache_name_is_unchanged(tmp_path):
    """The generalised cache key must still produce the exact legacy MB filename."""
    evaluator = _evaluator(n_samples=1000, seed=7, k=0, pred_coverage=True)
    evaluator.run_name = 'run'
    evaluator.checkpoint_prefix = 'best'
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    path = evaluator._summaryCachePath('test', 'mb', mask=None)

    assert path.name == 'summary_test_mb_run_best_s1000_seed7_k0_predcov1_all.pt'


def test_mb_imh_proposal_adds_refinement_time_per_dataset(monkeypatch, tmp_path):
    evaluator = _evaluator(
        rescale=False,
        n_samples=1000,
        seed=0,
        k=0,
        batch_size=8,
        likelihood_family=0,
    )
    evaluator.ckpt_dir = tmp_path / 'ckpt'
    evaluator.checkpoint_prefix = 'latest'
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test

    base = _proposal(tpd=0.5)
    refined = _proposal()
    seen = {}

    def fake_refine(method, base_proposal, batch, *args, **kwargs):
        seen['method'] = method
        seen['base'] = base_proposal
        seen['batch'] = batch
        return refined, 8.0

    monkeypatch.setattr(evaluator, '_loadOrSampleMb', lambda partition, dl: base)
    monkeypatch.setattr('metabeta.evaluation.evaluate.loadOrRefine', fake_refine)

    full_batch = {'X': torch.zeros(4, 3), 'sd_y': torch.ones(4)}
    proposal, mask = evaluator._getProposalAndMask('MB+IMH', 'test', full_batch, object())

    assert proposal is refined
    assert mask is None
    assert seen['method'] == 'imhMarginal'
    assert seen['base'] is base
    assert seen['batch'] is full_batch
    # MB sampling cost per dataset plus the refinement pass amortised over the batch
    assert proposal.tpd == pytest.approx(0.5 + 8.0 / 4)


def test_mb_imh_refines_in_rescaled_space(monkeypatch, tmp_path):
    evaluator = _evaluator(
        rescale=True,
        n_samples=1000,
        seed=0,
        k=0,
        batch_size=8,
        likelihood_family=0,
    )
    evaluator.ckpt_dir = tmp_path / 'ckpt'
    evaluator.checkpoint_prefix = 'latest'
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test
    seen = {}

    def fake_refine(method, base_proposal, batch, *args, **kwargs):
        seen['batch'] = batch
        return _proposal(), 0.0

    monkeypatch.setattr(evaluator, '_loadOrSampleMb', lambda partition, dl: _proposal(tpd=0.1))
    monkeypatch.setattr('metabeta.evaluation.evaluate.loadOrRefine', fake_refine)

    full_batch = {
        'X': torch.zeros(2, 3),
        'y': torch.ones(2, 5),
        'ffx': torch.ones(2, 1),
        'sd_y': torch.tensor([2.0, 4.0]),
    }
    evaluator._getProposalAndMask('MB+IMH', 'test', full_batch, object())

    # the batch handed to the sampler is rescaled, and the caller's batch is left untouched
    torch.testing.assert_close(seen['batch']['y'], torch.tensor([[2.0] * 5, [4.0] * 5]))
    torch.testing.assert_close(full_batch['y'], torch.ones(2, 5))


def test_mb_imh_appears_in_light_path_plot_labels(monkeypatch, tmp_path):
    evaluator = _evaluator(
        plot=True,
        converged_subset=False,
        rescale=False,
        n_samples=1000,
        seed=0,
        k=0,
        warmup=False,
        pred_coverage=False,
        summary_chunk_size=2,
        likelihood_family=0,
    )
    evaluator.data_path_test = tmp_path / 'test.fit.npz'
    evaluator.data_path_valid = evaluator.data_path_test
    evaluator.plot_dir = tmp_path / 'plots'
    evaluator.results_dir = None
    batch = {
        'X': torch.zeros(1, 1),
        'y': torch.zeros(1, 1),
        'ffx': torch.zeros(1, 1),
        'sigma_rfx': torch.ones(1, 1),
        'rfx': torch.zeros(1, 1, 1),
        'mask_d': torch.ones(1, 1, dtype=torch.bool),
        'mask_q': torch.ones(1, 1, dtype=torch.bool),
        'mask_mq': torch.ones(1, 1, 1, dtype=torch.bool),
        'sd_y': torch.ones(1),
    }
    labels_seen = []
    methods_seen = []

    monkeypatch.setattr(
        evaluator,
        '_getPartitionData',
        lambda *args, **kwargs: (object(), batch, tmp_path / 'test.npz'),
    )
    monkeypatch.setattr(evaluator, '_fitMaskFromPath', lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, '_fitProposalFromNpz', lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluator, '_getProposalAndMask', lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr(
        evaluator,
        '_loadCachedSummary',
        lambda partition, method, mask=None: methods_seen.append(method) or None,
    )
    monkeypatch.setattr(
        evaluator,
        '_loadOrComputeSummary',
        lambda proposal, batch, partition, method, mask=None: _lightSummary(),
    )
    monkeypatch.setattr(evaluator, '_alignToCommon', lambda proposal, *args: proposal)
    monkeypatch.setattr(
        evaluator,
        'plot',
        lambda proposals, summaries, labels, batch, plot_dir=None: labels_seen.extend(labels),
    )

    evaluator._evalPartitionLight(
        'test',
        ['MB', 'MB+IMH', 'NUTS'],
        fit_label='ppR2',
        multi=False,
    )

    assert labels_seen == ['MB', 'MB+IMH', 'NUTS']
    # MB+IMH looks up its own summary cache, not MB's
    assert methods_seen == ['mb', 'imhMarginal']


def _lightSummary():
    from metabeta.utils.evaluation import AggregatedMetrics, EvaluationSummary, PerDatasetMetrics

    return EvaluationSummary(
        per_dataset=PerDatasetMetrics(
            posterior_nll=torch.tensor([1.0]),
            loo_nll=torch.tensor([1.5]),
            pp_fit=torch.tensor([0.1]),
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
        tpd=1.0,
    )


def test_mb_imh_rejected_without_checkpoint():
    """Data-direct mode has no checkpoint, so MB+IMH must be refused up front."""
    import sys
    from metabeta.evaluation.evaluate import setup

    argv = [
        'evaluate.py',
        '--data_path_test',
        'test.fit.npz',
        '--models',
        'MB+IMH,NUTS',
    ]
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(ValueError, match='no MB / MB\\+IMH without --checkpoint'):
            setup()
    finally:
        sys.argv = old


def test_common_mask_alignment_handles_unmasked_mb_imh():
    proposal = _proposal()
    common = np.array([True, False])

    aligned = Evaluator._alignToCommon(proposal, None, common)

    assert aligned.samples_g.shape[0] == 1
