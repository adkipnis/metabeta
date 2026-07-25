import torch
import pytest

from metabeta.evaluation.cache import _fitBatchMask, _parseMethods, _subsetBatch


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
