import pytest
import torch

from metabeta.evaluation.cache import (
    _buildProposal,
    _fitBatchMask,
    _mergeSummaries,
    _parseMethods,
    _subsetBatch,
)
from metabeta.utils.evaluation import AggregatedMetrics, EvaluationSummary, PerDatasetMetrics


def test_parse_methods_accepts_comma_separated_methods():
    assert _parseMethods('nuts, advi,laplace') == ('nuts', 'advi', 'laplace')


def test_parse_methods_rejects_unknown_method():
    with pytest.raises(ValueError, match='unknown cache method'):
        _parseMethods('nuts,bogus')


def test_fit_batch_mask_keeps_all_without_failed_key():
    batch = {'y': torch.zeros(3, 2)}

    mask = _fitBatchMask(batch, 'nuts')

    assert mask.tolist() == [True, True, True]


def test_fit_batch_mask_drops_failed_rows():
    batch = {
        'y': torch.zeros(4, 2),
        'advi_failed': torch.tensor([False, True, False, True]),
    }

    mask = _fitBatchMask(batch, 'advi')

    assert mask.tolist() == [True, False, True, False]


def test_subset_batch_only_indexes_dataset_axis_tensors():
    batch = {
        'y': torch.arange(12).reshape(4, 3),
        'advi_ffx': torch.arange(8).reshape(4, 2),
        'mask_d': torch.ones(4, 2, dtype=torch.bool),
        'constant': torch.arange(3),
        'label': 'unchanged',
    }
    mask = torch.tensor([True, False, True, False])

    out = _subsetBatch(batch, mask)

    assert out['y'].shape == (2, 3)
    assert out['advi_ffx'].shape == (2, 2)
    assert out['mask_d'].shape == (2, 2)
    assert torch.equal(out['constant'], batch['constant'])
    assert out['label'] == 'unchanged'


def test_build_proposal_sets_time_per_dataset_from_duration():
    batch = {
        'laplace_ffx': torch.zeros(2, 3, 1),
        'laplace_sigma_rfx': torch.ones(2, 3, 1),
        'laplace_rfx': torch.zeros(2, 4, 3, 1),
        'laplace_duration': torch.tensor([0.1, 0.3]),
    }

    proposal = _buildProposal(batch, 'laplace', d_corr=0)

    assert proposal.tpd == pytest.approx(0.2)


def _summary(values: list[float]) -> EvaluationSummary:
    t = torch.tensor(values)
    return EvaluationSummary(
        per_dataset=PerDatasetMetrics(
            posterior_nll=t,
            loo_nll=t + 10.0,
            pp_fit=t + 20.0,
        ),
        aggregated=AggregatedMetrics(
            corr={},
            nrmse={},
            coverage={0.1: {'ffx': torch.ones(1)}},
            ece={},
            eace={},
            lcr={},
            abs_lcr={},
            estimates={
                'ffx': t.unsqueeze(-1),
                'rfx': t.view(-1, 1, 1),
            },
        ),
    )


def test_merge_summaries_restores_original_dataset_order(monkeypatch):
    monkeypatch.setattr('metabeta.evaluation.cache.getCorrelation', lambda *args, **kwargs: {})
    monkeypatch.setattr('metabeta.evaluation.cache.getRMSE', lambda *args, **kwargs: {})

    merged = _mergeSummaries(
        partials=[_summary([30.0, 10.0]), _summary([20.0])],
        small_data_list=[
            {
                'ffx': torch.tensor([[30.0], [10.0]]),
                'rfx': torch.tensor([[[30.0]], [[10.0]]]),
                'mask_d': torch.ones(2, 1, dtype=torch.bool),
            },
            {
                'ffx': torch.tensor([[20.0]]),
                'rfx': torch.tensor([[[20.0]]]),
                'mask_d': torch.ones(1, 1, dtype=torch.bool),
            },
        ],
        batch_sizes=[2, 1],
        dataset_indices=[torch.tensor([2, 0]), torch.tensor([1])],
        likelihood_family=1,
        all_rfx_ranks=[],
    )

    torch.testing.assert_close(
        merged.per_dataset.posterior_nll,
        torch.tensor([10.0, 20.0, 30.0]),
    )
    torch.testing.assert_close(
        merged.per_dataset.loo_nll,
        torch.tensor([20.0, 30.0, 40.0]),
    )
    torch.testing.assert_close(
        merged.aggregated.estimates['ffx'].squeeze(-1),
        torch.tensor([10.0, 20.0, 30.0]),
    )
