import argparse
import yaml
from typing import cast
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

from metabeta.simulation import hypersample, Prior, Synthesizer, Emulator, Simulator
from metabeta.utils.io import datasetFilename
from metabeta.utils.sampling import truncLogUni
from metabeta.utils.padding import aggregate


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hierarchical datasets.')
    # batch dimensions
    parser.add_argument('--bs_train', type=int, default=4096, help='batch size per training partition (default = 4,096).')
    parser.add_argument('--bs_valid', type=int, default=256, help='batch size for validation partition (default = 256).')
    parser.add_argument('--bs_test', type=int, default=128, help='batch size per testing partition (default = 128).')
    parser.add_argument('--bs_mini', type=int, default=32, help='training minibatch size (for grouping m, q, d - default = 32)')
    # partitions and sources
    parser.add_argument('--d_tag', type=str, default='toy', help='name of data config file')
    parser.add_argument('--partition', type=str, default='all', help='Type of partition in [train, valid, test, all], (default = train)')
    parser.add_argument('-b', '--begin', type=int, default=1, help='Begin generating training epoch number #b.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Total number of training epochs to generate.')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD if ds_type==sampled (default = False)')
    parser.add_argument('--loop', action='store_true', help='Loop dataset sampling instead of parallelizing it with joblib (default = False)')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Generator class

@dataclass
class Generator:
    cfg: argparse.Namespace
    outdir: Path = Path('..', 'outputs', 'data')

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        data_cfg_path = Path('configs', f'{cfg.d_tag}.yaml')
        assert data_cfg_path.exists(), f'config file {data_cfg_path} does not exist'
        with open(data_cfg_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            for k, v in data_cfg.items():
                setattr(self.cfg, k, v)

    def _genSizes(
        self,
        rng: np.random.Generator,
        n_datasets: int,
        mini_batch_size: int,
    ) -> tuple[np.ndarray, ...]:
        ''' batch generate size arrays for dataset sampling '''
        assert n_datasets % mini_batch_size == 0, 'number of datasets must be divisible by mini batch size'
        n_mini_batches = n_datasets // mini_batch_size

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
        return d, q, m, ns


    @staticmethod
    def _genDataset(
        cfg: argparse.Namespace, seedseq: np.random.SeedSequence, d: int, q: int, ns: np.ndarray,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seedseq)

        # sample prior
        hyperparams = hypersample(rng, d, q)
        prior = Prior(rng, hyperparams)

        # instantiate design
        if cfg.ds_type in ['toy', 'flat']:
            design = Synthesizer(rng, toy=(cfg.ds_type == 'toy'))
        elif cfg.ds_type == 'sampled':
            design = Emulator(rng, source=cfg.source, use_sgld=cfg.sgld)
        else:
            raise NotImplementedError(f'design sampler type {cfg.ds_type} is not implemented')

        # sample from simulator
        sim = Simulator(rng, prior, design, ns)
        return sim.sample()


    def _genBatch(
        self, n_datasets: int, mini_batch_size: int, epoch: int = 0,
    ) -> list[dict[str, np.ndarray]]:
        ''' generate list of {n_datasets} and keep m, d, q constant per minibatch '''
        # --- init seeding
        main_seed = {'train': epoch, 'valid': 10_000, 'test': 20_000}[self.cfg.partition]
        rng = np.random.default_rng(main_seed)
        seedseqs = np.random.SeedSequence(main_seed).spawn(n_datasets)
        desc = f'{epoch:02d}/{self.cfg.epochs:02d}' if self.cfg.partition == 'train' else ''

        # --- presample sizes
        d, q, m, ns = self._genSizes(rng, n_datasets, mini_batch_size)

        # --- sample batch of single datasets
        if self.cfg.loop: # Option A: loop
            datasets = []
            for i in tqdm(range(n_datasets), desc=desc):
                dataset = self._genDataset(
                    self.cfg,
                    seedseqs[i],
                    d[i],
                    q[i],
                    ns[i][: m[i]],
                )
                datasets.append(dataset)
        else: # Option B: parallelize
            datasets = Parallel(n_jobs=-1, backend='loky', batch_size='auto')(
                delayed(Generator._genDataset)(
                    self.cfg,
                    seedseqs[i],
                    d[i],
                    q[i],
                    ns[i][: m[i]],
                )
                for i in tqdm(range(n_datasets), desc=desc)
            )
            datasets = cast(list[dict[str, np.ndarray]], datasets) # joblib returns list[Any]
        return datasets


    def genTrain(self):
        assert self.cfg.begin > 0, 'starting training partition must be a positive integer'
        assert self.cfg.begin <= self.cfg.epochs, 'starting epoch larger than goal epoch'
        assert self.cfg.epochs < 10_000, 'maximum number of epochs exceeded'
        assert self.cfg.ds_type != 'sampled', 'training data must be synthetic'
        print(f'Generating {self.cfg.epochs} training partitions of {self.cfg.bs_train} datasets each...')
        for epoch in range(self.cfg.begin, self.cfg.epochs + 1):
            ds_train = self._genBatch(n_datasets=self.cfg.bs_train, mini_batch_size=self.cfg.bs_mini, epoch=epoch)
            ds_train = aggregate(ds_train)
            fn = Path(self.outdir, datasetFilename(self.cfg, 'train', epoch))
            np.savez_compressed(fn, **ds_train, allow_pickle=True)
            print(f'Saved training set to {fn}')

    def genValid(self):
        print('Generating validation set...')
        ds_valid = self._genBatch(n_datasets=self.cfg.bs_valid, mini_batch_size=1)
        ds_valid = aggregate(ds_valid)
        fn = Path(self.outdir, datasetFilename(self.cfg, 'valid'))
        np.savez_compressed(fn, **ds_valid, allow_pickle=True)
        print(f'Saved validation set to {fn}')

    def genTest(self):
        print('Generating test set...')
        ds_test = self._genBatch(n_datasets=self.cfg.bs_test, mini_batch_size=1)
        ds_test = aggregate(ds_test)
        fn = Path(self.outdir, datasetFilename(self.cfg, 'test'))
        np.savez_compressed(fn, **ds_test, allow_pickle=True)
        print(f'Saved test set to {fn}')

    def go(self):
        if self.cfg.partition == 'test':
            self.genTest()
        elif self.cfg.partition == 'valid':
            self.genValid()
        elif self.cfg.partition == 'train':
            self.genTrain()
        elif self.cfg.partition == 'all':
            self.cfg.partition = 'test'
            self.genTest()
            self.cfg.partition = 'valid'
            self.genValid()
            self.cfg.partition = 'train'
            self.genTrain()
        else:
            raise NotImplementedError(
                f'the partition type must be in [train, valid, test], but is {self.cfg.partition}')


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = setup()
    generator = Generator(cfg)
    generator.go()

