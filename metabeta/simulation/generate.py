import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np

from metabeta.simulation import hypersample, Prior, Synthesizer, Emulator, Simulator
from metabeta.utils.io import datasetFilename
from metabeta.utils.sampling import truncLogUni


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hierarchical datasets.')
    # batch dimensions
    parser.add_argument('--bs_train', type=int, default=4096, help='batch size per training partition (default = 4,096).')
    parser.add_argument('--bs_val', type=int, default=256, help='batch size for validation partition (default = 256).')
    parser.add_argument('--bs_test', type=int, default=128, help='batch size per testing partition (default = 128).')
    parser.add_argument('--bs_load', type=int, default=16, help='Batch size when loading (for grouping m, q, d, default = 16)')
    # data dimensions
    parser.add_argument('-d', '--max_d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 16).')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='Maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).')
    parser.add_argument('--min_m', type=int, default=5, help='Minimum number of groups (default = 5).')
    parser.add_argument('--max_m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='Minimum number of samples per group (default = 10).')
    parser.add_argument('--max_n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    # partitions and sources
    parser.add_argument('--partition', type=str, default='val', help='Type of partition in [train, val, test], (default = train)')
    parser.add_argument('-b', '--begin', type=int, default=1, help='Begin generating training epoch number #b.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Total number of training epochs to generate.')
    parser.add_argument('--type', type=str, default='toy', help='Type of predictors [toy, flat, scm, sampled], (default = toy)')
    parser.add_argument('--source', type=str, default='all', help='Source dataset if type==sampled (default = all)')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD if type==sampled (default = False)')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Generator class

@dataclass
class Generator:
    cfg: argparse.Namespace
    outdir: Path

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _maxShapes(batch: list[dict[str, np.ndarray]]) -> dict[str, tuple[int, ...]]:
        ''' for each array in dataset, get the maximal shape over the whole batch '''
        out = {}
        for dataset in batch:
            for key, array in dataset.items():
                if not isinstance(array, np.ndarray):
                    raise ValueError('expected all entries to be arrays')
                shape = tuple(array.shape)
                if key not in out:
                    out[key] = shape
                    continue
                assert len(shape) == len(out[key]), 'ndim mismatch'
                # Expand max_shapes[key] tuple elementwise to max dimension
                out[key] = tuple(
                    max(old_dim, new_dim)
                    for old_dim, new_dim in zip(out[key], shape)
                )
        return out


    def _aggregate(self, batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        ''' collate list of datasets to single batched dataset
            note: assumes consistency in keys and dtypes in batch '''
        out = {}
        max_shapes = self._maxShapes(batch)
        batch_size = len(batch)

        # init with zeros
        for key, shape in max_shapes.items():
            dtype = batch[0][key].dtype
            out[key] = np.zeros((batch_size, *shape), dtype=dtype)

        # fill with slicing
        for i, dataset in enumerate(batch):
            for key, dest in out.items():
                src = dataset[key]
                slc = (i, *tuple(slice(0, s) for s in src.shape))
                dest[slc] = src
        return out


    def _genDataset(
        self, rng: np.random.Generator, d: int, q: int, ns: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # sample prior
        hyperparams = hypersample(rng, d, q)
        prior = Prior(rng, hyperparams)

        # instantiate design
        if self.cfg.type in ['toy', 'flat']:
            design = Synthesizer(rng, toy=(self.cfg.type == 'toy'))
        elif self.cfg.type in ['sampled']:
            design = Emulator(rng, source=self.cfg.source, use_sgld=self.cfg.sgld)
        else:
            raise NotImplementedError(f'design sampler type {self.cfg.type} is not implemented')

        # instantiate simulator
        sim = Simulator(rng, prior, design, ns)
        return sim.sample()


