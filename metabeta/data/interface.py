from pathlib import Path
import numpy as np
import torch

DATA_DIR = Path('real')

class RealDataset:
    def __init__(
        self,
        source: Path | dict,
    ):
        # load
        if isinstance(source, Path):
            assert source.exists(), f'file {source} does not exist'
            np_dat = np.load(source, allow_pickle=True)
            self.original = {k: np_dat[k] for k in np_dat.files}
        elif isinstance(source, dict):
            self.original = source
        else:
            raise TypeError(f'source should be either Path or dict but is {type(source)}')

        # check if data is pre-grouped
        is_grouped = self.original['groups'] is not None
        assert is_grouped, 'ungrouped data not supported, yet'

        # store arrays as tensors
        self.data = {
            k: torch.from_numpy(v)
            for k, v in self.original.items()
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        }

    def __len__(self):
        return self.data['n']

