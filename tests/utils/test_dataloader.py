from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from metabeta.utils.dataloader import Collection, Dataloader


@pytest.fixture(scope='session')
def dataset_path() -> Path:
    # same file as in your __main__ block
    fname = 'valid_d3_q1_m5-30_n10-70_toy.npz'
    path = Path('metabeta', 'outputs', 'data', fname)
    assert path.exists(), f'{path} does not exist (tests expect the demo file to be present)'
    return path


def test_collection_loads_and_repr(dataset_path: Path):
    col = Collection(dataset_path, permute=True)
    assert len(col) > 0
    rep = repr(col)
    assert rep.startswith('Collection(')
    assert 'max(fixed)=' in rep
    assert 'max(random)=' in rep


def test_getitem_shapes_and_permutations(dataset_path: Path):
    col = Collection(dataset_path, permute=True)
    ds = col[0]

    for key in ('X', 'Z', 'y', 'ns', 'm', 'n', 'd', 'q'):
        assert key in ds

    n = int(ds['n'])
    m = int(ds['m'])
    d_max = int(col.d)
    q_max = int(col.q)

    assert ds['X'].shape == (n, d_max)
    assert ds['Z'].shape == (n, q_max)
    assert ds['y'].shape == (n,)

    # ns is max-padded for the collator
    assert ds['ns'].ndim == 1
    assert int(ds['ns'][:m].sum()) == n
    assert (ds['ns'][m:] == 0).all()

    # permutations exist when permute=True
    assert 'dperm' in ds
    assert 'qperm' in ds
    assert ds['dperm'].shape == (d_max,)
    assert ds['qperm'].shape == (q_max,)

    # permutation sanity: a true permutation of [0..d_max-1] / [0..q_max-1]
    assert np.array_equal(np.sort(ds['dperm']), np.arange(d_max))
    assert np.array_equal(np.sort(ds['qperm']), np.arange(q_max))

    # intercept fixed (assumed by your samplePermutation)
    assert int(ds['dperm'][0]) == 0
    assert int(ds['qperm'][0]) == 0


def test_dataloader_wrapper_shapes_and_dtypes(dataset_path: Path):
    bs = 4
    dl = Dataloader(dataset_path, batch_size=bs)
    batch = next(iter(dl))

    assert isinstance(batch, dict)

    for key in ('X', 'Z', 'y', 'ns', 'mask_n', 'mask_m', 'mask_d', 'mask_q', 'rfx'):
        assert key in batch
        assert isinstance(batch[key], torch.Tensor)

    # shapes
    assert batch['X'].ndim == 4  # (B, m_max, n_i_max, d_max)
    assert batch['Z'].ndim == 4  # (B, m_max, n_i_max, q_max)
    assert batch['y'].ndim == 3  # (B, m_max, n_i_max)

    B, m_max, n_i_max, d_max = batch['X'].shape
    assert B == bs
    assert batch['Z'].shape[:3] == (B, m_max, n_i_max)
    assert batch['y'].shape == (B, m_max, n_i_max)
    assert batch['ns'].shape == (B, m_max)
    assert batch['rfx'].shape == (B, m_max, batch['Z'].shape[-1])

    # dtypes
    assert batch['X'].dtype == torch.float32
    assert batch['Z'].dtype == torch.float32
    assert batch['y'].dtype == torch.float32
    assert batch['ns'].dtype == torch.int64
    assert batch['mask_n'].dtype == torch.bool
    assert batch['mask_m'].dtype == torch.bool
    assert batch['mask_d'].dtype == torch.bool
    assert batch['mask_q'].dtype == torch.bool


def test_mask_n_matches_ns(dataset_path: Path):
    dl = Dataloader(dataset_path, batch_size=4)
    batch = next(iter(dl))

    ns = batch['ns'].to(torch.int64)
    n_i_max = batch['y'].shape[-1]
    t = torch.arange(n_i_max, dtype=torch.int64)[None, None, :]
    expected = t < ns[:, :, None]

    assert torch.equal(batch['mask_n'], expected)


def test_mask_m_matches_ns_nonzero(dataset_path: Path):
    dl = Dataloader(dataset_path, batch_size=4)
    batch = next(iter(dl))

    expected = batch['ns'] != 0
    assert torch.equal(batch['mask_m'], expected)


def test_mask_dq_are_rowwise_permuted(dataset_path: Path):
    # This checks that mask_d/mask_q correspond to 'd'/'q' sizes *after* applying dperm/qperm.
    col = Collection(dataset_path, permute=True)

    # build a small batch deterministically
    batch_np = [col[i] for i in range(min(4, len(col)))]

    # use the public wrapper to collate (through iteration)
    dl = Dataloader(dataset_path, batch_size=len(batch_np))
    batch = next(iter(dl))

    d_max = batch['mask_d'].shape[-1]
    q_max = batch['mask_q'].shape[-1]

    # reconstruct expected masks from each sample's d/q and permutation
    exp_d = torch.zeros_like(batch['mask_d'])
    exp_q = torch.zeros_like(batch['mask_q'])

    for b, ds in enumerate(batch_np):
        d_i = int(ds['d'])
        q_i = int(ds['q'])

        base_d = np.zeros((d_max,), dtype=bool)
        base_d[:d_i] = True
        base_q = np.zeros((q_max,), dtype=bool)
        base_q[:q_i] = True

        if 'dperm' in ds:
            base_d = base_d[np.asarray(ds['dperm'], dtype=int)]
        if 'qperm' in ds:
            base_q = base_q[np.asarray(ds['qperm'], dtype=int)]

        exp_d[b] = torch.as_tensor(base_d, dtype=torch.bool)
        exp_q[b] = torch.as_tensor(base_q, dtype=torch.bool)

    assert torch.equal(batch['mask_d'], exp_d)
    assert torch.equal(batch['mask_q'], exp_q)


def test_dataloader_repr(dataset_path: Path):
    dl = Dataloader(dataset_path, batch_size=8)
    rep = repr(dl)
    assert rep.startswith('Dataloader(')
    assert 'batch_size=8' in rep
    assert 'Collection(' in rep
