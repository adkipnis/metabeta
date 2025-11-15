from pathlib import Path
import itertools
import numpy as np
import torch
from metabeta.data.dataset import split, getCollater, inversePermutation

DATA_DIR = Path('real')


class RealDataset:
    def __init__(
        self,
        data: dict | None = None,
        path: Path | None = None,
    ):
        # load and store as tensors
        if data is None:
            assert path is not None, 'must provide either data or path'
            assert path.exists(), (
                f'File {path} does not exist, you must generate it first using generate.py'
            )
            np_dat = np.load(path, allow_pickle=True)
            data = {k: np_dat[k] for k in np_dat.files}
            del np_dat

        # sizes
        if data['n_i'] is not None:
            self.max_n_i = int(max(data['n_i']))

        # default priors
        self.nu_ffx = torch.zeros(self.d)
        self.tau_ffx = torch.ones(self.d) * 50.0
        self.tau_ffx[0] *= 2  # intercept prior
        self.tau_rfx = torch.ones(self.d) * 30.0
        self.tau_eps = torch.tensor(10.0)

        # make data compliant for model
        self.original = data
        data = {
            k: torch.from_numpy(v)
            for k, v in data.items()
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        }
        self.data_long = self.curate(data)
        self.data = self.deepen(self.data_long)

        # save all possible permutations for workers
        perms = torch.tensor(list(itertools.permutations(range(d - 1)))) + 1
        zeros = torch.zeros((len(perms), 1))
        self.perms = torch.cat([zeros, perms], dim=-1).int()
        self.unperms = torch.stack([inversePermutation(p) for p in self.perms])

    def __len__(self):
        return self.len

    def addIntercept(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x[..., 0:1])
        return torch.cat([ones, x], dim=-1)

    def curate(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # sizes
        d = data['d']
        self.d = int(max(d, self.d))
        q = torch.tensor(1)  # this should be dynamic
        n = data['n']
        m = data['m']
        n_i = data['n_i']

        # inputs
        y = data['y'].float()
        X = data['X']
        X = self.addIntercept(X).float()
        Z = X.clone()
        Z[..., q:] = 0
        groups = data['groups']

        # mask priors
        self.tau_ffx[d:] = 0.0
        self.tau_rfx[q:] = 0.0

        # output
        out = {
            # sizes
            'm': m,
            'n': n,
            'n_i': n_i,
            'd': d,
            'q': q,
            # inputs
            'y': y,
            'X': X,
            'Z': Z,
            'groups': groups,
            # priors
            'nu_ffx': self.nu_ffx,
            'tau_ffx': self.tau_ffx,
            'tau_eps': self.tau_eps,
            'tau_rfx': self.tau_rfx,
        }
        return out

    def deepen(self, ds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ds = {k: v.clone() for k, v in ds.items()}
        y, X, Z = ds['y'], ds['X'], ds['Z']
        n_i = ds['n_i']
        m = int(ds['m'])
        d = ds['d']
        q = ds['q']

        y = split(y, n_i, shape=[m, self.max_n_i])
        X = split(X, n_i, shape=[m, self.max_n_i, self.d])
        Z = split(Z, n_i, shape=[m, self.max_n_i, self.d])

        # mask
        mask_d = torch.ones(self.d, dtype=torch.bool)
        mask_d[d:] = False
        mask_n = y != 0.0
        mask_m = torch.ones(m, dtype=torch.bool)
        mask_q = torch.ones(self.d, dtype=torch.bool)
        mask_q[q:] = False
        non_empty = torch.ones(self.d, dtype=torch.bool)
        non_empty[self.q :] = False

        out = {
            # inputs
            'y': y,
            'X': X,
            'Z': Z,
            # masks
            'mask_d': mask_d,
            'mask_q': mask_q,
            'mask_n': mask_n,
            'mask_m': mask_m,
            'non_empty': non_empty,
        }
        ds.update(**out)
        return ds

    def permute(
        self, ds: dict[str, torch.Tensor], perm: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        ds = {k: v.clone() for k, v in ds.items()}
        ds['X'] = ds['X'][..., perm]
        ds['Z'] = ds['Z'][..., perm]
        ds['nu_ffx'] = ds['nu_ffx'][..., perm]
        ds['tau_ffx'] = ds['tau_ffx'][..., perm]
        ds['tau_rfx'] = ds['tau_rfx'][..., perm]
        ds['mask_d'] = ds['mask_d'][..., perm]
        ds['mask_q'] = ds['mask_q'][..., perm]
        return ds

    def raw(self):
        return self.data_long

    def batch(self, n_workers: int = 8, device: str = 'cpu') -> dict[str, torch.Tensor]:
        collate_fn = getCollater(device=device)
        if n_workers > len(self.perms):
            print(
                f'Only {len(self.perms)} unique permutations are possible, reducing n_workers.'
            )
            n_workers = len(self.perms)
        out = []
        for perm, unperm in zip(self.perms[:n_workers], self.unperms[:n_workers]):
            ds = self.permute(self.data, perm)
            mask = ds['non_empty'][perm]
            ds['Z'] = ds['Z'][..., mask]
            ds['tau_rfx'] = ds['tau_rfx'][..., mask]
            ds['mask_q'] = ds['mask_q'][..., mask]
            ds['perm'] = perm
            ds['unperm'] = unperm
            out += [ds]
        return collate_fn(out)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    path = Path(DATA_DIR, 'math.npz')
    rds = RealDataset(path=path)
    ds_long = rds.data_long
    ds = rds.batch()

