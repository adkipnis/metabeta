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

