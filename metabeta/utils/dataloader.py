from pathlib import Path
import numpy as np
import torch

from metabeta.utils.sampling import samplePermutation
from metabeta.utils.padding import unpad, padToModel


def toDevice(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    # make sure device is torch.device
    if isinstance(device, str):
        device = torch.device(device)

    # stop if devices already match
    current_device = None
    for k, v in batch.items():
        if torch.is_tensor(v):
            current_device = v.device
            break
    if (current_device is None) or (current_device == device):
        return batch

    # cast each tensor to the desired device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


class Collection(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        permute: bool = True,
        max_d: int | None = None,
        max_q: int | None = None,
    ):
        super().__init__()

        # load data
        assert path.exists(), f'{path} does not exist'
        with np.load(path, allow_pickle=True) as raw:
            self.raw = dict(raw)

        # reduce host-memory footprint for training/validation collections
        # by storing floating arrays in float32 and large integers in int32.
        for key, value in self.raw.items():
            if not isinstance(value, np.ndarray):
                continue
            if value.dtype.kind == 'f' and value.dtype != np.float32:
                self.raw[key] = value.astype(np.float32)
            elif (
                value.dtype.kind in ('i', 'u')
                and value.dtype.itemsize > np.dtype(np.int32).itemsize
            ):
                lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
                v_min = value.min()
                v_max = value.max()
                if lo <= v_min and v_max <= hi:
                    self.raw[key] = value.astype(np.int32)

        self.has_params = 'ffx' in self.raw
        self.has_nuts = 'nuts_ffx' in self.raw
        self.has_advi = 'advi_ffx' in self.raw

        # quickly assert that group indices are ascending from 0 to m-1
        self._groupCheck(len(self))

        # shapes
        d_file = int(self.raw['d'].max())  # fixed effects in file
        q_file = int(self.raw['q'].max())  # random effects in file
        if max_d is not None and max_d < d_file:
            raise ValueError(
                f'max_d override ({max_d}) is smaller than file maximum d ({d_file}) for {path}'
            )
        if max_q is not None and max_q < q_file:
            raise ValueError(
                f'max_q override ({max_q}) is smaller than file maximum q ({q_file}) for {path}'
            )
        self.d = int(max_d) if max_d is not None else d_file
        self.q = int(max_q) if max_q is not None else q_file
        self.m_i = self.raw['m'].astype(int, copy=False)
        self.n_i_max = self.raw['ns'].max(axis=1).astype(int, copy=False)

        # feature permutations
        self.permute = permute and self.has_params
        if self.permute:
            rng = np.random.default_rng(0)
            self.dperm = [samplePermutation(rng, self.d) for _ in range(len(self))]
            self.qperm = [samplePermutation(rng, self.q) for _ in range(len(self))]

    def __len__(self) -> int:
        return len(self.raw['y'])

    def _groupCheck(self, n_datasets: int = 8):
        """quick sanity check that group indices are contiguous"""
        n_datasets = min(n_datasets, len(self))
        checks = np.zeros((n_datasets,), dtype=bool)
        for i in range(n_datasets):
            m = self.raw['m'][i]
            n = self.raw['n'][i]
            ns = self.raw['ns'][i].astype(int, copy=False)
            g = self.raw['groups'][i, :n].astype(int, copy=False)
            diffs = np.diff(g)
            ascending = (0 <= diffs).all() and (diffs <= 1).all()
            correct_borders = g[0] == 0 and g[-1] == m - 1
            sums_to_n = ns.sum() == n
            ns_padded = (ns[m:] == 0).all()
            checks[i] = ascending and correct_borders and sums_to_n and ns_padded
        assert checks.all(), 'group indices are not structured correctly'

    def __repr__(self) -> str:
        return f'Collection({len(self)} datasets, max(fixed)={self.d}, max(random)={self.q}, fits={self.has_nuts or self.has_advi})'

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # get dataset (without fit statistics)
        ds = {
            k: v[idx]
            for k, v in self.raw.items()
            # if not (k.startswith('nuts') or k.startswith('advi'))
        }

        # unpad to actual d/q, then pad to collection/model maxima
        sizes = {k: ds[k] for k in ('m', 'n')}
        sizes['d'] = int(ds['d'])
        sizes['q'] = int(ds['q'])
        ds = unpad(ds, sizes)
        ds = padToModel(ds, max_d=self.d, max_q=self.q)

        # padToModel builds Z from padded X and zeros inactive random-effect columns.

        # optionally permute
        if self.permute:
            # fixed effects and related
            dperm = self.dperm[idx]
            for key in ('X', 'ffx', 'nu_ffx', 'tau_ffx'):
                ds[key] = ds[key][..., dperm]
            ds['dperm'] = dperm

            # random effects and related
            qperm = self.qperm[idx]
            for key in ('Z', 'rfx', 'sigma_rfx', 'tau_rfx'):
                ds[key] = ds[key][..., qperm]
            ds['corr_rfx'] = ds['corr_rfx'][np.ix_(qperm, qperm)]
            ds['qperm'] = qperm

            # permute fit samples consistently with ground truth
            for method in ('nuts', 'advi'):
                if f'{method}_ffx' in ds:
                    ds[f'{method}_ffx'] = ds[f'{method}_ffx'][dperm]
                    ds[f'{method}_sigma_rfx'] = ds[f'{method}_sigma_rfx'][qperm]
                    ds[f'{method}_rfx'] = ds[f'{method}_rfx'][qperm]
                    if f'{method}_corr_rfx' in ds:
                        ds[f'{method}_corr_rfx'] = ds[f'{method}_corr_rfx'][..., qperm, :][..., qperm]

        return ds


def quickCollate(batch: list[dict[str, np.ndarray]], key: str, dtype=torch.float32) -> torch.Tensor:
    tensors = [torch.as_tensor(ds[key], dtype=dtype) for ds in batch]
    return torch.stack(tensors, dim=0)


def collateGrouped(
    batch: list[dict[str, np.ndarray]], dtype=torch.float32
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    B = len(batch)
    m = int(max(ds['m'] for ds in batch))
    n_i = int(max(max(ds['ns'][: ds['m']]) for ds in batch))
    d = int(batch[0]['X'].shape[-1])  # X is max-padded in last dim
    q = int(batch[0]['Z'].shape[-1])  # same for Z

    # helpers for deepening
    d_ = np.arange(d)
    q_ = np.arange(q)
    n_i_ = torch.arange(n_i).unsqueeze(0)

    # deepen (batch, obs, feat) -> (batch, group, group_obs, feat)
    y = torch.zeros((B, m, n_i), dtype=dtype)
    X = torch.zeros((B, m, n_i, d), dtype=dtype)
    Z = torch.zeros((B, m, n_i, q), dtype=dtype)
    mask_n = torch.zeros((B, m, n_i), dtype=torch.bool)
    mask_d = torch.zeros((B, d), dtype=torch.bool)
    mask_q = torch.zeros((B, q), dtype=torch.bool)
    ns = torch.zeros((B, m), dtype=torch.int64)
    for b, ds in enumerate(batch):
        mask_d[b] = torch.as_tensor(d_ < ds['d'])
        mask_q[b] = torch.as_tensor(q_ < ds['q'])
        ns[b, : ds['m']] = torch.as_tensor(ds['ns'])
        idx = torch.as_tensor(n_i_ < ns[b].unsqueeze(-1))
        y[b, idx] = torch.as_tensor(ds['y'], dtype=dtype)
        X[b, idx] = torch.as_tensor(ds['X'], dtype=dtype)
        Z[b, idx] = torch.as_tensor(ds['Z'], dtype=dtype)
        mask_n[b] = idx
    out.update(
        {
            'X': X,
            'Z': Z,
            'y': y,
            'ns': ns,
            'mask_d': mask_d,
            'mask_q': mask_q,
            'mask_n': mask_n,
        }
    )

    # cast integers to long tensors
    out['n'] = quickCollate(batch, 'n', torch.int64)
    out['m'] = quickCollate(batch, 'm', torch.int64)

    # cast params to float tensors
    for key in (
        'ffx',
        'sigma_rfx',
        'nu_ffx',
        'tau_ffx',
        'tau_rfx',
    ):
        out[key] = quickCollate(batch, key, dtype)
    for key in ('sigma_eps', 'tau_eps'):
        if key in batch[0]:
            out[key] = quickCollate(batch, key, dtype)

    # extra treatment for rfx due to m dimension
    rfx = torch.zeros((B, m, q), dtype=dtype)
    for b, ds in enumerate(batch):
        idx = torch.as_tensor(np.arange(m) < ds['m'])
        rfx[b, idx] = torch.as_tensor(ds['rfx'], dtype=dtype)
    out['rfx'] = rfx

    # correlation matrix (q, q) → (B, q, q); eta_rfx=0 means identity
    out['corr_rfx'] = quickCollate(batch, 'corr_rfx', dtype)
    out['eta_rfx'] = quickCollate(batch, 'eta_rfx', dtype)

    # likelihood and prior family indices
    if 'likelihood_family' in batch[0]:
        out['likelihood_family'] = quickCollate(batch, 'likelihood_family', torch.long)
    for key in ('family_ffx', 'family_sigma_rfx', 'family_sigma_eps'):
        if key in batch[0]:
            out[key] = quickCollate(batch, key, torch.long)

    # save sd(Y) for unstandardizing
    out['sd_y'] = quickCollate(batch, 'sd_y')

    # remaining mask handling
    out['mask_m'] = out['ns'] != 0
    if 'dperm' in batch[0]:
        out['dperm'] = quickCollate(batch, 'dperm', torch.int64)
        out['qperm'] = quickCollate(batch, 'qperm', torch.int64)
        out['mask_d'] = torch.gather(out['mask_d'], dim=1, index=out['dperm'])
        out['mask_q'] = torch.gather(out['mask_q'], dim=1, index=out['qperm'])

    # mask_mq and mask_corr must be built after any mask_q permutation handling.
    mq = out['mask_q']
    out['mask_mq'] = out['mask_m'].unsqueeze(-1) & mq.unsqueeze(-2)
    q = mq.shape[-1]
    out['mask_corr'] = torch.stack(
        [mq[..., i] & mq[..., j] for i in range(1, q) for j in range(i)], dim=-1
    ) if q >= 2 else mq.new_zeros(B, 0)

    # collate fit samples if present
    for method in ('nuts', 'advi'):
        if f'{method}_ffx' in batch[0]:
            out.update(collateFits(batch, method, d, q, m, dtype))

    return out


def collateFits(
    batch: list[dict[str, np.ndarray]],
    method: str,
    d: int,
    q: int,
    m: int,
    dtype=torch.float32,
) -> dict[str, torch.Tensor]:
    B = len(batch)
    s = batch[0][f'{method}_ffx'].shape[-1]
    has_eps = f'{method}_sigma_eps' in batch[0]
    out: dict[str, torch.Tensor] = {}

    ffx = torch.zeros((B, s, d), dtype=dtype)
    sigma_rfx = torch.zeros((B, s, q), dtype=dtype)
    rfx = torch.zeros((B, m, s, q), dtype=dtype)
    if has_eps:
        sigma_eps = torch.zeros((B, s), dtype=dtype)

    for b, ds in enumerate(batch):
        # (d, s) -> (s, d)
        ffx[b] = torch.as_tensor(ds[f'{method}_ffx'], dtype=dtype).T
        # (q, s) -> (s, q)
        sigma_rfx[b] = torch.as_tensor(ds[f'{method}_sigma_rfx'], dtype=dtype).T
        # (1, s) -> (s,)
        if has_eps:
            sigma_eps[b] = torch.as_tensor(ds[f'{method}_sigma_eps'], dtype=dtype).squeeze(0)
        # (q, m_i, s) -> (m_i, s, q)
        rfx_i = torch.as_tensor(ds[f'{method}_rfx'], dtype=dtype).permute(1, 2, 0)
        m_i = int(ds['m'])
        max_m = min(m_i, rfx_i.shape[0])
        if max_m > 0:
            rfx[b, :max_m] = rfx_i[:max_m]

    out[f'{method}_ffx'] = ffx
    out[f'{method}_sigma_rfx'] = sigma_rfx
    out[f'{method}_rfx'] = rfx
    if has_eps:
        out[f'{method}_sigma_eps'] = sigma_eps

    if f'{method}_corr_rfx' in batch[0]:
        corr_rfx = torch.zeros((B, s, q, q), dtype=dtype)
        for b, ds in enumerate(batch):
            c = torch.as_tensor(ds[f'{method}_corr_rfx'], dtype=dtype).squeeze(0)  # (s, q_i, q_i)
            q_i = c.shape[-1]
            corr_rfx[b, :, :q_i, :q_i] = c
        out[f'{method}_corr_rfx'] = corr_rfx

    out[f'{method}_duration'] = quickCollate(batch, f'{method}_duration')

    # per-dataset diagnostics: shape (b, chains) or (b, n_params)
    for diag_key in (
        f'{method}_divergences',
        f'{method}_max_treedepth',
        f'{method}_ess',
        f'{method}_ess_tail',
        f'{method}_rhat',
    ):
        if diag_key in batch[0]:
            out[diag_key] = quickCollate(batch, diag_key)

    return out


def subsetBatch(
    batch: dict[str, torch.Tensor], mask: np.ndarray
) -> dict[str, torch.Tensor]:
    """Filter a collated batch dict to the datasets selected by a boolean mask."""
    idx = torch.from_numpy(mask)
    return {
        k: v[idx] if torch.is_tensor(v) and v.shape[0] == len(mask) else v
        for k, v in batch.items()
    }


class SortishBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch sampler that sorts collection to reduce memory demands"""

    def __init__(
        self,
        m_i: np.ndarray,
        n_i_max: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        bucket_mult: int = 50,
        seed: int = 0,
    ):
        self.m_i = m_i
        self.n_i_max = n_i_max
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.bucket_mult = max(1, int(bucket_mult))
        self.seed = int(seed)
        self.n = len(m_i)

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(self.n)
        if self.shuffle:
            idx = rng.permutation(idx)

        bucket_size = min(self.n, self.batch_size * self.bucket_mult)
        ordered: list[int] = []
        for start in range(0, self.n, bucket_size):
            chunk = idx[start : start + bucket_size]
            # sort by m first (primary), then n_i_max (secondary) — this minimises
            # the padded tensor area because m varies more widely than n_i_max and
            # area-based sorting groups datasets with the same product but different
            # shapes, causing cross-axis waste.
            order = np.lexsort((self.n_i_max[chunk], self.m_i[chunk]))
            ordered.extend(chunk[order].tolist())

        batches = [
            ordered[start : start + self.batch_size]
            for start in range(0, len(ordered), self.batch_size)
        ]
        if self.shuffle and len(batches) > 1:
            batch_order = rng.permutation(len(batches))
            batches = [batches[i] for i in batch_order]

        for batch in batches:
            yield batch


class Dataloader(torch.utils.data.DataLoader):
    """Wrapper for torch dataloader"""

    def __init__(
        self,
        path: Path,
        batch_size: int | None = None,
        sortish: bool = True,
        shuffle: bool = False,
        bucket_mult: int = 50,
        sort_seed: int = 0,
        max_d: int | None = None,
        max_q: int | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        col = Collection(path, max_d=max_d, max_q=max_q)
        pin_memory = torch.cuda.is_available()
        self._sortish = sortish
        self._shuffle = shuffle
        self._bucket_mult = bucket_mult
        self._sort_seed = sort_seed
        if batch_size is None:
            batch_size = len(col)
        else:
            batch_size = min(batch_size, len(col))
        self._batch_size_effective = batch_size

        use_sortish = sortish and batch_size < len(col)
        persistent = num_workers > 0 and persistent_workers
        if use_sortish:
            batch_sampler = SortishBatchSampler(
                m_i=col.m_i,
                n_i_max=col.n_i_max,
                batch_size=batch_size,
                shuffle=shuffle,
                bucket_mult=bucket_mult,
                seed=sort_seed,
            )
            super().__init__(
                dataset=col,
                batch_sampler=batch_sampler,
                pin_memory=pin_memory,
                collate_fn=collateGrouped,
                num_workers=num_workers,
                persistent_workers=persistent,
            )
        else:
            super().__init__(
                dataset=col,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                collate_fn=collateGrouped,
                num_workers=num_workers,
                persistent_workers=persistent,
            )

    def __repr__(self) -> str:
        return f'Dataloader(batch_size={self._batch_size_effective}) for {self.dataset}'

    def fullBatch(self) -> dict[str, torch.Tensor]:
        col = self.dataset
        n = len(col)
        if n == 0:
            raise ValueError('cannot collate empty dataset')

        use_sortish = self._sortish and self._batch_size_effective < n
        if use_sortish:
            batch_sampler = SortishBatchSampler(
                m_i=col.m_i,
                n_i_max=col.n_i_max,
                batch_size=self._batch_size_effective,
                shuffle=self._shuffle,
                bucket_mult=self._bucket_mult,
                seed=self._sort_seed,
            )
            idx = [i for batch in batch_sampler for i in batch]
        else:
            idx = list(range(n))
            if self._shuffle:
                rng = np.random.default_rng(self._sort_seed)
                idx = rng.permutation(idx).tolist()

        batch = [col[i] for i in idx]
        return collateGrouped(batch)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    from metabeta.utils.config import dataFromYaml

    # load toy data
    data_cfg_path = Path('..', 'simulation', 'configs', 'small.yaml')
    data_fname = dataFromYaml(data_cfg_path, 'test')
    data_path = Path('..', 'outputs', 'data', data_fname)

    # get dataset collection
    col = Collection(data_path)
    print(col)

    # get dataloader
    dl = Dataloader(data_path, batch_size=8)
    minibatch = next(iter(dl))

    # print batch shapes
    for k, v in minibatch.items():
        print(f'{k}: {v.numpy().shape}')
    print(dl)
