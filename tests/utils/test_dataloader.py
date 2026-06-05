from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from metabeta.utils.dataloader import Collection, Dataloader


@pytest.fixture(scope='session')
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp('dataloader-data') / 'test.npz'
    rng = np.random.default_rng(42)

    n_datasets = 8
    max_d = 4
    max_q = 2
    max_m = 5
    max_n = 32
    d_vals = np.array([2, 3, 4, 2, 3, 4, 2, 4], dtype=np.int64)
    q_vals = np.array([1, 1, 2, 1, 2, 2, 1, 2], dtype=np.int64)
    m_vals = np.array([3, 4, 5, 3, 4, 5, 3, 4], dtype=np.int64)

    X = np.zeros((n_datasets, max_n, max_d), dtype=np.float32)
    y = np.zeros((n_datasets, max_n), dtype=np.float32)
    groups = np.zeros((n_datasets, max_n), dtype=np.int64)
    ns = np.zeros((n_datasets, max_m), dtype=np.int64)

    for i in range(n_datasets):
        m_i = int(m_vals[i])
        counts = rng.integers(2, 8, size=m_i, dtype=np.int64)
        n_i = int(counts.sum())
        ns[i, :m_i] = counts
        groups[i, :n_i] = np.repeat(np.arange(m_i), counts)
        X[i, :n_i, 0] = 1.0
        X[i, :n_i, 1 : d_vals[i]] = rng.normal(size=(n_i, int(d_vals[i]) - 1))
        y[i, :n_i] = rng.normal(size=n_i)

    n = ns.sum(axis=1).astype(np.int64)
    np.savez(
        path,
        ffx=np.zeros((n_datasets, max_d), dtype=np.float32),
        sigma_rfx=np.ones((n_datasets, max_q), dtype=np.float32),
        sigma_eps=np.ones(n_datasets, dtype=np.float32),
        corr_rfx=np.repeat(np.eye(max_q, dtype=np.float32)[None, :, :], n_datasets, axis=0),
        rfx=np.zeros((n_datasets, max_m, max_q), dtype=np.float32),
        likelihood_family=np.zeros(n_datasets, dtype=np.int64),
        nu_ffx=np.zeros((n_datasets, max_d), dtype=np.float32),
        tau_ffx=np.ones((n_datasets, max_d), dtype=np.float32),
        tau_rfx=np.ones((n_datasets, max_q), dtype=np.float32),
        tau_eps=np.ones(n_datasets, dtype=np.float32),
        eta_rfx=np.zeros(n_datasets, dtype=np.float32),
        family_ffx=np.zeros(n_datasets, dtype=np.int64),
        family_sigma_rfx=np.zeros(n_datasets, dtype=np.int64),
        family_sigma_eps=np.zeros(n_datasets, dtype=np.int64),
        y=y,
        X=X,
        groups=groups,
        m=m_vals,
        n=n,
        ns=ns,
        d=d_vals,
        q=q_vals,
        sd_y=np.ones(n_datasets, dtype=np.float32),
    )
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

    # build a small batch deterministically — sortish=False keeps items in index order
    # so col[i] and dl's iteration agree on which item occupies position i
    batch_np = [col[i] for i in range(min(4, len(col)))]

    # use the public wrapper to collate (through iteration)
    dl = Dataloader(dataset_path, batch_size=len(batch_np), sortish=False, permute=True)
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

    # qperm must equal the first q elements of dperm (within-group invariant)
    for ds in batch_np:
        if 'dperm' in ds:
            np.testing.assert_array_equal(ds['dperm'][: len(ds['qperm'])], ds['qperm'])

    # mask_mq and mask_corr must be derived from the final (possibly permuted) mask_q.
    expected_mq = batch['mask_m'].unsqueeze(-1) & batch['mask_q'].unsqueeze(-2)
    assert torch.equal(batch['mask_mq'], expected_mq)

    mq = batch['mask_q']
    q = mq.shape[-1]
    if q >= 2:
        expected_corr = torch.stack(
            [mq[..., i] & mq[..., j] for i in range(1, q) for j in range(i)], dim=-1
        )
        assert torch.equal(batch['mask_corr'], expected_corr)
    else:
        assert batch['mask_corr'].shape[-1] == 0


def test_dataloader_repr(dataset_path: Path):
    dl = Dataloader(dataset_path, batch_size=8)
    rep = repr(dl)
    assert rep.startswith('Dataloader(')
    assert 'batch_size=8' in rep
    assert 'Collection(' in rep


def test_dataloader_respects_explicit_max_dims(dataset_path: Path):
    col_default = Collection(dataset_path, permute=True)
    d_override = int(col_default.d) + 2
    q_override = int(col_default.q) + 2

    col = Collection(dataset_path, permute=True, max_d=d_override, max_q=q_override)
    assert col.d == d_override
    assert col.q == q_override

    dl = Dataloader(dataset_path, batch_size=4, sortish=False, max_d=d_override, max_q=q_override)
    batch = next(iter(dl))
    assert batch['X'].shape[-1] == d_override
    assert batch['Z'].shape[-1] == q_override
    assert batch['mask_d'].shape[-1] == d_override
    assert batch['mask_q'].shape[-1] == q_override


def test_collection_rejects_too_small_overrides(dataset_path: Path):
    col_default = Collection(dataset_path, permute=True)
    with pytest.raises(ValueError, match='max_d override'):
        Collection(dataset_path, permute=True, max_d=int(col_default.d) - 1)
    with pytest.raises(ValueError, match='max_q override'):
        Collection(dataset_path, permute=True, max_q=int(col_default.q) - 1)
