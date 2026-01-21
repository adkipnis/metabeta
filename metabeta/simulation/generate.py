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
    parser.add_argument('--type', type=str, default='sampled', help='Type of predictors [toy, flat, scm, sampled], (default = toy)')
    parser.add_argument('--source', type=str, default='all', help='Source dataset if type==sampled (default = all)')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD if type==sampled (default = False)')
    parser.add_argument('--loop', action='store_true', help='Loop dataset sampling instead of parallelizing it with joblib (default = False)')
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


    def _genBatch(
        self, n_datasets: int, mini_batch_size: int, epoch: int = 0,
    ) -> list[dict[str, np.ndarray]]:
        ''' generate list of {n_datasets} keep m, d, q constant per minibatch '''
        assert n_datasets % mini_batch_size == 0, 'number of datasets must be divisible by mini batch size'
        n_mini_batches = n_datasets // mini_batch_size

        # --- init
        iterator = tqdm(range(n_datasets))
        if self.cfg.partition == 'test':
            iterator.set_description(f'{epoch:02d}/{self.cfg.epochs:02d}')
        seed = {'train': epoch, 'val': 10_000, 'test': 20_000}[self.cfg.partition]
        rng = np.random.default_rng(seed)
        datasets = []

        # --- presample sizes
        # number of fixed effects
        d = rng.integers(low=2, high=self.cfg.max_d+1, size=n_mini_batches)
        d = np.repeat(d, mini_batch_size)

        # number of random effects
        q = truncLogUni(rng, low=1, high=self.cfg.max_q+1, size=n_mini_batches, round=True)
        q = np.repeat(q, mini_batch_size)
        q = np.minimum(d, q) # q <= d

        # number of groups
        m = truncLogUni(rng, low=self.cfg.min_m, high=self.cfg.max_m+1, size=n_mini_batches, round=True)
        m = np.repeat(m, mini_batch_size)

        # number of observations per group
        ns = truncLogUni(rng, low=self.cfg.min_n, high=self.cfg.max_n+1, size=(n_datasets, self.cfg.max_m), round=True)
    @staticmethod
    def _genDataset(
        cfg: argparse.Namespace, seedseq: np.random.SeedSequence, d: int, q: int, ns: np.ndarray,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seedseq)

        # sample prior
        hyperparams = hypersample(rng, d, q)
        prior = Prior(rng, hyperparams)

        # instantiate design
        if cfg.type in ['toy', 'flat']:
            design = Synthesizer(rng, toy=(cfg.type == 'toy'))
        elif cfg.type == 'sampled':
            design = Emulator(rng, source=cfg.source, use_sgld=cfg.sgld)
        else:
            raise NotImplementedError(f'design sampler type {cfg.type} is not implemented')

        # sample from simulator
        sim = Simulator(rng, prior, design, ns)
        return sim.sample()

        # --- loop over single datasets
        for i in iterator:
            d_i, q_i, m_i = d[i], q[i], m[i]
            ns_i = ns[i][:m_i]
            dataset = self._genDataset(rng, d_i, q_i, ns_i)
            datasets.append(dataset)
        return datasets


    def genTrain(self):
        assert self.cfg.begin > 0, 'starting training partition must be a positive integer'
        assert self.cfg.begin <= self.cfg.epochs, 'starting epoch larger than goal epoch'
        assert self.cfg.epochs < 10_000, 'maximum number of epochs exceeded'
        assert self.cfg.type != 'sampled', 'training data must be synthetic'
        print(f'Generating {self.cfg.epochs} training partitions of {self.cfg.bs_train} datasets each...')
        for epoch in range(self.cfg.begin, self.cfg.epochs + 1):
            ds_train = self._genBatch(n_datasets=self.cfg.bs_train, mini_batch_size=self.cfg.bs_load, epoch=epoch)
            ds_train = self._aggregate(ds_train)
            fn = Path(self.outdir, datasetFilename(self.cfg, epoch))
            np.savez_compressed(fn, **ds_train, allow_pickle=True)
            print(f'Saved training set to {fn}')

    def genVal(self):
        print('Generating validation set...')
        ds_val = self._genBatch(n_datasets=self.cfg.bs_val, mini_batch_size=1)
        ds_val = self._aggregate(ds_val)
        fn = Path(self.outdir, datasetFilename(self.cfg))
        np.savez_compressed(fn, **ds_val, allow_pickle=True)
        print(f'Saved validation set to {fn}')

    def genTest(self):
        print('Generating test set...')
        ds_test = self._genBatch(n_datasets=self.cfg.bs_test, mini_batch_size=1)
        ds_test = self._aggregate(ds_test)
        fn = Path(self.outdir, datasetFilename(self.cfg))
        np.savez_compressed(fn, **ds_test, allow_pickle=True)
        print(f'Saved test set to {fn}')

    def go(self):
        if self.cfg.partition == 'train':
            self.genTrain()
        elif self.cfg.partition == 'val':
            self.genVal()
        elif self.cfg.partition == 'test':
            self.genTest()
        else:
            raise NotImplementedError(
                f'the partition type must be in [train, val, test], but is {self.cfg.partition}')


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = setup()
    outdir = Path('..', 'outputs', 'data')
    generator = Generator(cfg, outdir)
    generator.go()

