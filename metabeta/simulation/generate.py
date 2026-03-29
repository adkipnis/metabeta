import logging
import argparse
import yaml
from typing import cast
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

from metabeta.simulation import (
    hypersample,
    Prior,
    Synthesizer,
    Scammer,
    Emulator,
    Simulator,
)
from metabeta.utils.io import datasetFilename
from metabeta.utils.sampling import truncLogUni
from metabeta.utils.padding import aggregate

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# config
# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hierarchical datasets.')
    # batch dimensions
    parser.add_argument('--bs_train', type=int, default=4096, help='batch size per training partition (default = 4,096).')
    parser.add_argument('--bs_valid', type=int, default=256, help='batch size for validation partition (default = 256).')
    parser.add_argument('--bs_test', type=int, default=128, help='batch size per testing partition (default = 128).')
    parser.add_argument('--bs_mini', type=int, default=32, help='training minibatch size (for grouping m, q, d - default = 32)')
    # partitions and sources
    parser.add_argument('--d_tag', type=str, default='small-p-sampled', help='name of data config file')
    parser.add_argument('--partition', type=str, default='train', help='Type of partition in [train, valid, test, all], (default = train)')
    parser.add_argument('-b', '--begin', type=int, default=1, help='Begin generating training epoch number #b.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Total number of training epochs to generate.')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD if ds_type==sampled (default = False)')
    parser.add_argument('--loop', action='store_true', help='Loop dataset sampling instead of parallelizing it with joblib (default = False)')
    return parser.parse_args()
# fmt: on


# -----------------------------------------------------------------------------
# Generator class


@dataclass
class Generator:
    cfg: argparse.Namespace
    outdir: Path = Path(__file__).resolve().parent / '..' / 'outputs' / 'data'

    # size-regime defaults (m = groups, n = obs per group)
    M_CAP: int = 300
    N_TOTAL_CAP: int = 5000
    NPG_MIN: int = 2
    REGIME_PROBS: tuple[float, float, float] = (0.3, 0.5, 0.2)
    # (m_low, m_high, npg_low, npg_high)
    REGIMES: tuple[tuple[int, int, int, int], ...] = (
        (80, 300, 2, 8),
        (20, 120, 5, 25),
        (8, 40, 20, 60),
    )

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        data_cfg_path = Path(__file__).resolve().parent / 'configs' / f'{self.cfg.d_tag}.yaml'
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
        """batch generate size arrays for dataset sampling"""
        assert (
            n_datasets % mini_batch_size == 0
        ), 'number of datasets must be divisible by mini batch size'
        n_mini_batches = n_datasets // mini_batch_size

        # --- presample dimensions
        # number of fixed effects
        d = rng.integers(low=2, high=self.cfg.max_d + 1, size=n_mini_batches)
        d = np.repeat(d, mini_batch_size)

        # number of random effects
        q = truncLogUni(rng, low=1, high=self.cfg.max_q + 1, size=n_mini_batches, round=True)
        q = np.repeat(q, mini_batch_size)
        q = np.minimum(d, q)  # q <= d

        # --- regime-mixture size sampling
        regime_ids = rng.choice(
            len(self.REGIMES),
            size=n_mini_batches,
            p=np.array(self.REGIME_PROBS, dtype=float),
        )
        regime_ids = np.repeat(regime_ids, mini_batch_size)

        m = np.zeros(n_datasets, dtype=int)
        ns = np.zeros((n_datasets, self.M_CAP), dtype=int)
        for i in range(n_datasets):
            m_low, m_high, npg_low, npg_high = self.REGIMES[int(regime_ids[i])]

            m_i = int(rng.integers(m_low, m_high + 1))
            m_i = min(m_i, self.M_CAP)

            npg_target = float(rng.uniform(npg_low, npg_high))
            n_total = int(round(m_i * npg_target * rng.uniform(0.85, 1.15)))
            n_total = max(n_total, m_i * self.NPG_MIN)
            n_total = min(n_total, self.N_TOTAL_CAP)
            n_total = max(n_total, m_i * self.NPG_MIN)

            ns_i = self._sampleCountsBounded(
                rng=rng,
                n=n_total,
                m=m_i,
                min_n=self.NPG_MIN,
                max_n=npg_high,
            )

            m[i] = m_i
            ns[i, :m_i] = ns_i
        return d, q, m, ns

    @staticmethod
    def _sampleCountsBounded(
        rng: np.random.Generator,
        n: int,
        m: int,
        min_n: int,
        max_n: int,
    ) -> np.ndarray:
        if n < m * min_n:
            raise ValueError(f'n={n} too small for m={m} and min_n={min_n}')
        if n > m * max_n:
            n = m * max_n

        ns = np.full(m, min_n, dtype=int)
        rem = n - int(ns.sum())
        if rem == 0:
            return ns

        caps = np.full(m, max_n - min_n, dtype=int)
        while rem > 0:
            alpha = rng.uniform(2.0, 20.0)
            p = rng.dirichlet(np.ones(m) * alpha)
            extra = rng.multinomial(rem, p)
            extra = np.minimum(extra, caps)
            added = int(extra.sum())
            if added == 0:
                idx = np.where(caps > 0)[0]
                if len(idx) == 0:
                    break
                j = int(rng.choice(idx))
                extra[j] = 1
                added = 1
            ns += extra
            caps -= extra
            rem -= added
        assert rem == 0, 'failed to allocate all observations under bounds'
        return ns

    @staticmethod
    def _genDataset(
        cfg: argparse.Namespace,
        seedseq: np.random.SeedSequence,
        d: int,
        q: int,
        ns: np.ndarray,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seedseq)

        # sample prior
        likelihood_family = getattr(cfg, 'likelihood_family', 0)
        hyperparams = hypersample(rng, d, q, likelihood_family=likelihood_family)
        prior = Prior(rng, hyperparams)

        # instantiate design
        ds_type = cfg.ds_type
        if ds_type == 'mixed':
            ds_type = rng.choice(['flat', 'sampled', 'scm'])

        if ds_type in ['toy', 'flat']:
            design = Synthesizer(rng, toy=(ds_type == 'toy'))
        elif ds_type == 'scm':
            design = Scammer(rng)
        elif ds_type == 'sampled':
            design = Emulator(
                rng,
                source=cfg.source,
                use_sgld=cfg.sgld,
                min_m=Generator.REGIMES[-1][0],
                min_n=Generator.NPG_MIN,
                max_n=max(reg[3] for reg in Generator.REGIMES),
            )
        else:
            raise NotImplementedError(f'design sampler type {ds_type} is not implemented')

        # sample from simulator
        sim = Simulator(rng, prior, design, ns)
        return sim.sample()

    def _genBatch(
        self, n_datasets: int, mini_batch_size: int, epoch: int = 0
    ) -> list[dict[str, np.ndarray]]:
        """generate list of {n_datasets} and keep m, d, q constant per minibatch"""
        # --- init seeding
        main_seed = {'train': epoch, 'valid': 10_000, 'test': 20_000}[self.cfg.partition]
        rng = np.random.default_rng(main_seed)
        seedseqs = np.random.SeedSequence(main_seed).spawn(n_datasets)
        desc = ''
        if self.cfg.partition == 'train':
            desc = f'{epoch:02d}/{self.cfg.epochs:02d}'

        # --- presample sizes
        d, q, m, ns = self._genSizes(rng, n_datasets, mini_batch_size)

        # --- sample batch of single datasets
        if self.cfg.loop or self.cfg.ds_type in ['scm', 'mixed']:  # Option A: loop
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
        else:  # Option B: parallelize
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
            # joblib returns list[Any], so for type safety we cast it
            datasets = cast(list[dict[str, np.ndarray]], datasets)
        return datasets

    @staticmethod
    def _castCompactTypes(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in batch.items():
            if (
                isinstance(value, np.ndarray)
                and value.dtype.kind == 'f'
                and value.dtype != np.float32
            ):
                out[key] = value.astype(np.float32)
            elif (
                isinstance(value, np.ndarray)
                and value.dtype.kind in ('i', 'u')
                and value.dtype.itemsize > np.dtype(np.int32).itemsize
            ):
                lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
                v_min = value.min()
                v_max = value.max()
                if lo <= v_min and v_max <= hi:
                    out[key] = value.astype(np.int32)
                else:
                    out[key] = value
            else:
                out[key] = value
        return out

    def genTest(self):
        logger.info('Generating test set...')
        ds_test = self._genBatch(n_datasets=self.cfg.bs_test, mini_batch_size=1)
        ds_test = aggregate(ds_test)
        ds_test = self._castCompactTypes(ds_test)
        fn = Path(self.outdir, datasetFilename(vars(self.cfg), 'test'))
        np.savez_compressed(fn, **ds_test, allow_pickle=True)
        logger.info(f'Saved test set to {fn}')

    def genValid(self):
        logger.info('Generating validation set...')
        ds_valid = self._genBatch(n_datasets=self.cfg.bs_valid, mini_batch_size=1)
        ds_valid = aggregate(ds_valid)
        ds_valid = self._castCompactTypes(ds_valid)
        fn = Path(self.outdir, datasetFilename(vars(self.cfg), 'valid'))
        np.savez_compressed(fn, **ds_valid, allow_pickle=True)
        logger.info(f'Saved validation set to {fn}')

    def genTrain(self):
        assert self.cfg.begin > 0, 'starting training partition must be a positive integer'
        assert self.cfg.begin <= self.cfg.epochs, 'starting epoch larger than goal epoch'
        assert self.cfg.epochs < 10_000, 'maximum number of epochs exceeded'
        # assert self.cfg.ds_type != 'sampled', 'training data must be synthetic'
        logger.info(
            f'Generating {self.cfg.epochs} training partitions of {self.cfg.bs_train} datasets each...'
        )
        for epoch in range(self.cfg.begin, self.cfg.epochs + 1):
            ds_train = self._genBatch(
                n_datasets=self.cfg.bs_train,
                mini_batch_size=self.cfg.bs_mini,
                epoch=epoch,
            )
            ds_train = aggregate(ds_train)
            ds_train = self._castCompactTypes(ds_train)
            fn = Path(self.outdir, datasetFilename(vars(self.cfg), 'train', epoch))
            np.savez_compressed(fn, **ds_train, allow_pickle=True)
            logger.debug(f'Saved training set to {fn}')

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
                f'the partition type must be in [train, valid, test], but is {self.cfg.partition}'
            )


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = setup()
    generator = Generator(cfg)
    generator.go()
