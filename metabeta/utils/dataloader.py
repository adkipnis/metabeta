from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from metabeta.utils.sampling import samplePermutation
from metabeta.utils.padding import unpad


class Collection(Dataset):
    def __init__(
        self,
        path: Path,
        permute: bool = True,
    ):
        super().__init__()

        # load data
        assert path.exists(), f'{path} does not exist'
        with np.load(path, allow_pickle=True) as raw:
            self.raw = dict(raw)
        self.has_params = 'ffx' in self.raw

        # quickly assert that group indices are ascending from 0 to m-1
        self._groupCheck(len(self))

        # shapes
        self.d = int(self.raw['d'].max()) # fixed effects
        self.q = int(self.raw['q'].max()) # random effects

        # feature permutations
        self.permute = permute and self.has_params
        if self.permute:
            rng = np.random.default_rng(0)
            self.dperm = [samplePermutation(rng, self.d) for _ in range(len(self))]
            self.qperm = [samplePermutation(rng, self.q) for _ in range(len(self))]


    def __len__(self) -> int:
        return len(self.raw['y'])
 
    def _groupCheck(self, n_datasets: int = 8):
        ''' quick sanity check that group indices are contiguous '''
        n_datasets = min(n_datasets, len(self))
        checks = np.zeros((n_datasets,), dtype=bool)
        for i in range(n_datasets):
            m = self.raw['m'][i]
            n = self.raw['n'][i]
            ns = self.raw['ns'][i].astype(int, copy=False)
            g = self.raw['groups'][i, :n].astype(int, copy=False)
            diffs = np.diff(g)
            ascending = (0 <= diffs).all() and (diffs <= 1).all()
            correct_borders = (g[0] == 0 and g[-1] == m - 1)
            sums_to_n = (ns.sum() == n)
            ns_padded = (ns[m:] == 0).all()
            checks[i] = (ascending and correct_borders and sums_to_n and ns_padded)
        assert checks.all(), 'group indices are not structured correctly'

    def __repr__(self) -> str:
        return f'Collection({len(self)} datasets, max(fixed)={self.d}, max(random)={self.q})'

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # get dataset (without fit statistics)
        ds = {k: v[idx] for k,v in self.raw.items()
                        if not (k.startswith('nuts') or k.startswith('advi'))}
        ns = ds['ns'] # backup padded counts for use in collator
 
        # unpad m/n but keep d/q maximal
        sizes = {k: ds[k] for k in ('m', 'n')}
        sizes['d'] = self.d
        sizes['q'] = self.q
        ds = unpad(ds, sizes)

        # init rfx design matrix and re-insert max-padded ns
        ds['Z'] = ds['X'][..., : self.q].copy()
        ds['ns'] = ns

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
            ds['qperm'] = qperm

        return ds


def quickCollate(batch: list[dict[str, np.ndarray]], key: str, dtype=torch.float32) -> torch.Tensor:
    tensors = [torch.as_tensor(ds[key], dtype=dtype) for ds in batch]
    return torch.stack(tensors, dim=0)


def collateGrouped(batch: list[dict[str, np.ndarray]], dtype=torch.float32) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    B = len(batch)
    m = int(max(ds['m'] for ds in batch))
    n_i = int(max(max(ds['ns'][: ds['m']]) for ds in batch))
    d = int(batch[0]['X'].shape[-1]) # X is max-padded in last dim
    q = int(batch[0]['Z'].shape[-1]) # same for Z

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
        ns[b] = torch.as_tensor(ds['ns'][:m])
        idx = torch.as_tensor(n_i_ < ns[b].unsqueeze(-1))
        y[b, idx] = torch.as_tensor(ds['y'], dtype=dtype)
        X[b, idx] = torch.as_tensor(ds['X'], dtype=dtype)
        Z[b, idx] = torch.as_tensor(ds['Z'], dtype=dtype)
        mask_n[b] = idx
    out.update({'X': X, 'Z': Z, 'y': y, 'ns': ns,
                'mask_d': mask_d, 'mask_q': mask_q, 'mask_n': mask_n})

    # cast integer arrays to int tensors
    for key in ('ns', 'dperm', 'qperm'):
        if key in batch[0]:
            out[key] = quickCollate(batch, key, torch.int64)

    # cast params to float tensors
    for key in ('ffx', 'sigma_rfx', 'sigma_eps',
                'nu_ffx', 'tau_ffx', 'tau_rfx', 'tau_eps'):
        out[key] = quickCollate(batch, key, dtype)

    # extra treatment for rfx due to m dimension
    rfx = torch.zeros((B, m, q), dtype=dtype)
    for b, ds in enumerate(batch):
        idx = torch.as_tensor(np.arange(m) < ds['m'])
        rfx[b, idx] = torch.as_tensor(ds['rfx'], dtype=dtype)
    out['rfx'] = rfx
 
    # remaining mask handling
    out['mask_m'] = (out['ns'] != 0)
    if 'dperm' in out:
        out['mask_d'] = torch.gather(out['mask_d'], dim=1, index=out['dperm'])
        out['mask_q'] = torch.gather(out['mask_q'], dim=1, index=out['qperm'])
    return out


class Dataloader(DataLoader):
    ''' Wrapper for torch dataloader '''
    def __init__(self, path: Path, batch_size: int | None = 1,):
        not_mps = (torch.accelerator.current_accelerator().type != 'mps') # type: ignore
        super().__init__(
            dataset=Collection(path),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=not_mps,
            collate_fn=collateGrouped
        )

    def __repr__(self) -> str:
        return f'Dataloader(batch_size={self.batch_size}) for {self.dataset}'


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    fname = 'val_d3_q1_m5-30_n10-70_toy.npz'
    path = Path('..', 'outputs', 'data', fname)
    col = Collection(path)
    print(col)
    dl = Dataloader(path, batch_size=256)
    minibatch = next(iter(dl))
    for k,v in minibatch.items():
        print(f'{k}: {v.numpy().shape}')
    print(dl)
