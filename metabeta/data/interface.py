from pathlib import Path
import numpy as np
import torch


class RealDataset:
    def __init__(
        self,
        source: Path | dict,
    ):
        # load
        if isinstance(source, Path):
            assert source.exists(), 'source does not exist'
            np_dat = np.load(source, allow_pickle=True)
            self.original = {k: np_dat[k] for k in np_dat.files}
        elif isinstance(source, dict):
            self.original = source
        else:
            raise TypeError(f'source should be either Path or dict but is {type(source)}')

        # store arrays as tensors
        self.data = {
            k: torch.from_numpy(v)
            for k, v in self.original.items()
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        }
        
        # check dims
        assert len(self.data['X']) == self.data['n'], 'dim mismatch (X, n)'
        if 'groups' in self.data:
            assert len(self.data['groups']) == len(self.data['X']), 'dim mismatch (group, X)'
            assert len(self.data['groups'].unique()) == self.data['m'] , 'dim mismatch (group, m)'
            assert self.data['n'] == self.data['n_i'].sum(), 'dim mismatch (n, n_i)'
            
        # tag data
        if isinstance(source, Path):
            self.data['source'] = source

    def __len__(self):
        return self.data['n']


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    path = Path('real', 'preprocessed', 'math.npz')
    ds = RealDataset(path)
    