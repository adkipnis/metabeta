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
    bambiDefaultPriors,
    Prior,
    Synthesizer,
    Scammer,
    Emulator,
    Simulator,
)
from metabeta.simulation.emulator import Subsampler
from metabeta.utils.families import hasSigmaEps, LIKELIHOOD_FAMILIES
from metabeta.utils.io import datasetFilename
from metabeta.utils.sampling import truncLogUni
from metabeta.utils.padding import aggregate
from metabeta.utils.templates import setupConfigParser, generateSimulationConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# config
# fmt: off
def setup() -> argparse.Namespace:
    """Parse command line arguments.

    Usage modes
    -----------
    Fresh start (template-based):
        python generate.py --size small --family 0 --ds_type toy

        Generates config from size/family/ds_type presets and writes all
        partitions (train/valid/test) to outputs/data/{data_id}/. The data_id
        is auto-derived as '{size}-{family_char}-{ds_type}' (e.g. small-n-toy).

    Generate a single partition:
        python generate.py --size small --family 0 --ds_type toy --partition train
        python generate.py --size small --family 0 --ds_type toy --partition valid --bs_valid 512

    Resume / extend training epochs:
        python generate.py --size small --family 0 --ds_type toy --partition train -b 11 -e 30

        Generates epochs 11–30, leaving already-generated epochs untouched.

    Load config from a saved YAML:
        python generate.py --config outputs/data/small-n-toy/config.yaml --partition train -b 21 -e 40

        Reproduces the exact data dimensions from a previous run. Explicit CLI
        args (e.g. -b, -e, --partition) still override the loaded YAML.
    """
    parser = argparse.ArgumentParser(
        epilog='Data dimension overrides (max_d, max_q, min_m, max_m, min_n, max_n, max_n_total) can be set via --config.',
    )

    # Template-based config generation (primary interface)
    parser.add_argument('--size', type=str, default='small', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='sampled', help='Dataset type: toy|flat|scm|mixed|sampled|real')

    # Alternative: load config from a saved YAML (e.g. outputs/data/{data_id}/config.yaml)
    parser.add_argument('--config', type=str, help='Path to a saved config.yaml; explicit CLI args override its values')

    # Batch dimensions
    parser.add_argument('--bs_train', type=int, default=4096, help='Number of datasets per training epoch file (default = 4096)')
    parser.add_argument('--bs_valid', type=int, default=256, help='Number of datasets in the validation file (default = 256)')
    parser.add_argument('--bs_test', type=int, default=512, help='Number of datasets in the test file (default = 512); split into equal chunks for uncertainty estimates')
    parser.add_argument('--bs_mini', type=int, default=32, help='Mini-batch size for grouping m/q/d across datasets (default = 32)')

    # Partitions and sources
    parser.add_argument('--partition', type=str, default='all', help='Which partition(s) to generate: train|valid|test|eval|all (default = all)')
    parser.add_argument('-b', '--begin', type=int, default=1, help='First training epoch to generate (default = 1)')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Last training epoch to generate (default = 20)')
    parser.add_argument('--source', type=str, default='all', help='Dataset source key for sampled/real ds_type (default = all)')
    parser.add_argument('--sgld', action='store_true', help='Use SGLD sampler when ds_type=sampled (default = False)')
    parser.add_argument('--loop', action='store_false', help='Generate datasets sequentially instead of in parallel with joblib (default = True)')

    return setupConfigParser(parser, generateSimulationConfig, 'Generate hierarchical datasets.')
# fmt: on


# -----------------------------------------------------------------------------
# Generator class


@dataclass
class Generator:
    cfg: argparse.Namespace
    outdir: Path = Path(__file__).resolve().parent / '..' / 'outputs' / 'data'

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.max_m_feasible = min(self.cfg.max_m, self.cfg.max_n_total // self.cfg.min_n)
        if self.max_m_feasible < self.cfg.min_m:
            raise ValueError(
                'incompatible bounds: require min_m <= floor(max_n_total / min_n), '
                f'but got min_m={self.cfg.min_m}, min_n={self.cfg.min_n}, '
                f'max_n_total={self.cfg.max_n_total}'
            )

        self.max_d_feasible = self.cfg.max_d
        if self.cfg.ds_type == 'real':
            subsampler = Subsampler(
                np.random.default_rng(0),
                source=self.cfg.source,
                likelihood_family=self.cfg.likelihood_family,
                min_m=self.cfg.min_m,
                min_n=self.cfg.min_n,
                max_n=self.cfg.max_n,
            )
            self.max_d_feasible = min(self.cfg.max_d, subsampler.maxCompatibleD())
            min_d = 2 if getattr(self.cfg, 'partition', 'train') == 'train' else self.cfg.min_d
            if self.max_d_feasible < min_d:
                raise ValueError(
                    f'real source pool only supports max_d={self.max_d_feasible}, '
                    f'but requested minimum d is {min_d}'
                )
            self.cfg.max_d = self.max_d_feasible

    def _genDims(
        self,
        rng: np.random.Generator,
        n_datasets: int,
        mini_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample mini-batch-uniform d, q, m arrays."""
        assert (
            n_datasets % mini_batch_size == 0
        ), 'number of datasets must be divisible by mini batch size'
        n_mini = n_datasets // mini_batch_size

        # min_d/min_q define non-overlapping test bands; ignored during training
        # so that the single training run covers the full (d, q) space.
        # d and q are drawn log-uniformly over their bounded integer ranges.
        is_train = getattr(self.cfg, 'partition', 'train') == 'train'
        min_d = 2 if is_train else getattr(self.cfg, 'min_d', 2)
        max_d = self.max_d_feasible if self.cfg.ds_type == 'real' else self.cfg.max_d
        low_d, high_d = float(min_d), float(max_d + 1)
        d_uniq = truncLogUni(rng, low=low_d, high=high_d, size=n_mini, round=True).astype(int)
        d_uniq = d_uniq.clip(min_d, max_d)
        d = np.repeat(d_uniq, mini_batch_size)

        min_q = 1 if is_train else getattr(self.cfg, 'min_q', 1)
        q_max_i = np.minimum(self.cfg.max_q, d_uniq).astype(float)  # (n_mini,) upper bound
        q_hi = q_max_i + 1.0
        q_uniq = np.floor(
            np.exp(rng.uniform(np.log(float(min_q)), np.log(q_hi), size=n_mini))
        ).astype(int)
        q_uniq = q_uniq.clip(min_q, np.minimum(self.cfg.max_q, d_uniq))
        q = np.repeat(q_uniq, mini_batch_size)

        # Two identifiability constraints for variance-component estimation:
        #   (1) Fixed effects β: m > d  (m − d is the between-group df in
        #       Henderson Method 3; needed for the q=1 compacted path)
        #   (2) Random-effects covariance Ψ (q×q, q(q+1)/2 free parameters):
        #       m > q(q+1)/2 so there are enough group-level observations to
        #       identify Ψ regardless of likelihood family.
        # m_low = max(d, q(q+1)/2) + min_bg_df satisfies both simultaneously.
        # m is drawn from skewedBeta(m_low, max_m, mode=m_low+min(0.08*excess,20), c=8)
        # The absolute mode cap of 20 keeps the mode near real-data cluster regardless
        # of max_m (real test-set: p50≈19, p75≈71).  c=8 concentrates mass to the
        # left of the mode so the bulk of draws stay near small m even for huge presets.
        min_bg_df = getattr(self.cfg, 'min_bg_df', 0)
        psi_df = q_uniq * (q_uniq + 1) // 2  # free parameters in q×q Ψ
        m_low = np.minimum(
            np.maximum(self.cfg.min_m, np.maximum(d_uniq, psi_df) + min_bg_df),
            self.max_m_feasible,
        )
        # Inline skewedBeta(low=m_low, high=max_m+1, mode=m_low+min(0.08*excess,20), c=8)
        # vectorised over per-element m_low; mode is clamped so low < mode < high
        # even for the degenerate edge case m_low == max_m_feasible.
        high_m = float(self.max_m_feasible + 1)
        excess_m = (self.max_m_feasible - m_low).astype(float)
        m_mode = np.clip(
            m_low + np.minimum(0.08 * excess_m, 20.0),
            m_low + 1e-3,
            self.max_m_feasible - 1e-3,
        ).astype(float)
        t = (m_mode - m_low) / (high_m - m_low)   # fraction in (0, 1)
        a_beta = 1.0 + t * 7                        # concentration=8 → (c-1)=7
        b_beta = 1.0 + (1.0 - t) * 7
        m = np.floor(m_low + (high_m - m_low) * rng.beta(a_beta, b_beta)).astype(int)
        m = m.clip(m_low, self.max_m_feasible)
        m = np.repeat(m, mini_batch_size)

        return d, q, m

    def _genNs(
        self,
        rng: np.random.Generator,
        n_datasets: int,
        m: np.ndarray,
        min_ng: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample per-group observation counts with masking and max_n_total cap.

        min_ng: optional (n_datasets,) int array of per-dataset minimum group
                sizes.  When provided (e.g. from min_within_df), each group in
                dataset i gets at least min_ng[i] observations before capping.
        """
        effective_min_n = self.cfg.min_n if min_ng is None else int(np.max(min_ng))
        ns = truncLogUni(
            rng,
            low=self.cfg.min_n,
            high=self.cfg.max_n + 1,
            size=(n_datasets, self.cfg.max_m),
            round=True,
        )
        if min_ng is not None:
            ns = np.maximum(ns, min_ng[:, None])  # (n_datasets, max_m) per-dataset floor
        return self._maskAndCapNs(
            ns=ns,
            m=m,
            max_m=self.cfg.max_m,
            min_n=effective_min_n,
            max_n=self.cfg.max_n,
            max_n_total=self.cfg.max_n_total,
        )

    @staticmethod
    def _maskAndCapNs(
        ns: np.ndarray,
        m: np.ndarray,
        max_m: int,
        min_n: int,
        max_n: int,
        max_n_total: int,
    ) -> np.ndarray:
        idx = np.arange(max_m)
        active = idx[None, :] < m[:, None]
        ns = np.where(active, ns, 0)

        n_total = ns.sum(axis=1)
        n_total_min = m * min_n
        over = n_total > max_n_total
        if np.any(over):
            active_over = active[over]
            base = min_n * active_over.astype(int)
            extra = ns[over] - base
            target_extra = max_n_total - n_total_min[over]
            extra_total = extra.sum(axis=1)

            # smooth compression of extras to hit max_n_total without rejection sampling
            factor = target_extra / np.maximum(extra_total, 1)
            extra_f = extra * factor[:, None]
            extra_scaled = np.floor(extra_f).astype(int)

            cap_extra = max_n - min_n
            extra_scaled = np.clip(extra_scaled, 0, cap_extra)

            rem = target_extra - extra_scaled.sum(axis=1)
            if np.any(rem > 0):
                spare = cap_extra - extra_scaled
                residual = extra_f - extra_scaled
                residual = np.where((spare > 0) & active_over, residual, -1.0)

                order = np.argsort(-residual, axis=1)
                rank = np.empty_like(order)
                rows = np.arange(order.shape[0])[:, None]
                rank[rows, order] = np.arange(order.shape[1])[None, :]

                add = (rank < rem[:, None]).astype(int)
                add = np.minimum(add, spare)
                extra_scaled += add

            ns[over] = base + extra_scaled

        return ns

    @staticmethod
    def _genDataset(
        cfg: argparse.Namespace,
        seedseq: np.random.SeedSequence,
        d: int,
        q: int,
        ns: np.ndarray,
        min_n_eff: int = 0,  # per-dataset effective min_n (from min_within_df); 0 = use cfg.min_n
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seedseq)

        # sample prior
        likelihood_family = cfg.likelihood_family
        hyperparams = hypersample(rng, d, q, likelihood_family=likelihood_family)
        prior = Prior(rng, hyperparams)

        # instantiate design
        ds_type = cfg.ds_type
        if ds_type == 'mixed':
            ds_type = rng.choice(['flat', 'sampled', 'scm'])

        if ds_type == 'real':
            subsampler = Subsampler(
                rng,
                source=cfg.source,
                likelihood_family=likelihood_family,
                min_m=cfg.min_m,
                min_n=cfg.min_n,
                max_n=cfg.max_n,
            )
            obs = subsampler.sample(d, ns)

            # bambi-default hyperparameters
            m = len(obs['ns'])
            n = len(obs['y'])
            hyperparams = bambiDefaultPriors(d, q, likelihood_family)

            # NaN placeholders for ground-truth parameters
            out = {
                'ffx': np.full(d, np.nan),
                'sigma_rfx': np.full(q, np.nan),
                'corr_rfx': np.full((q, q), np.nan),
                'rfx': np.full((m, q), np.nan),
                **hyperparams,
                'y': obs['y'],
                'X': obs['X'],
                'groups': obs['groups'],
                'm': np.array(m),
                'n': np.array(n),
                'ns': obs['ns'],
                'd': np.array(d),
                'q': np.array(q),
                'sd_y': obs['sd_y'],
                'source': obs['source'],
            }
            if hasSigmaEps(likelihood_family):
                out['sigma_eps'] = np.array(np.nan)
                out['r_squared'] = np.array(np.nan)
            return out

        if ds_type in ['toy', 'flat']:
            design = Synthesizer(rng, toy=(ds_type == 'toy'))
        elif ds_type == 'scm':
            design = Scammer(rng)
        elif ds_type == 'sampled':
            design = Emulator(
                rng,
                source=cfg.source,
                use_sgld=getattr(cfg, 'sgld', False),
                min_m=cfg.min_m,
                min_n=max(cfg.min_n, min_n_eff),  # respect per-dataset min_within_df floor
                max_n=cfg.max_n,
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

        # --- presample dimensions (always mini-batch-uniform)
        d, q, m = self._genDims(rng, n_datasets, mini_batch_size)

        # --- presample per-group counts
        min_ng = None  # may be set in else branch
        if self.cfg.ds_type in ('sampled', 'real'):
            # Emulator/Subsampler override ns internally based on source dataset constraints;
            # only req_m = len(ns_i) and req_n = sum(ns_i) survive as loose hints.
            # Draw req_n the same way the flat/scm path does: sample a per-group n
            # log-uniformly from [min_n, max_n] and multiply by m.  This aligns the
            # req_n scale with what _genNs produces, so the Emulator receives a similar
            # total-n budget regardless of which sub-type mixed resolves to.
            n_hint_pg = truncLogUni(
                rng,
                low=self.cfg.min_n,
                high=self.cfg.max_n + 1,
                size=n_datasets,
                round=True,
            )
            n_hint = n_hint_pg * m
            n_hint = np.clip(
                n_hint, m * self.cfg.min_n, np.minimum(m * self.cfg.max_n, self.cfg.max_n_total)
            )
            ns_slices = [
                np.full(int(m[i]), int(n_hint[i]) // int(m[i]), dtype=int)
                for i in range(n_datasets)
            ]
        else:
            # toy, flat, scm, mixed: ns is used directly by Synthesizer/Scammer/Emulator
            # Enforce within-group df floor: n_g >= q + min_within_df so that ZtZ_g
            # has at least min_within_df residual df for per-group variance estimation.
            # Mirrors min_bg_df (between-group floor) but applied within each group.
            min_within_df = getattr(self.cfg, 'min_within_df', 0)
            min_ng = np.maximum(self.cfg.min_n, q + min_within_df) if min_within_df else None
            ns = self._genNs(rng, n_datasets, m, min_ng=min_ng)
            ns_slices = [ns[i][: m[i]] for i in range(n_datasets)]

        # --- sample batch of single datasets
        min_n_effs = (
            [int(min_ng[i]) for i in range(n_datasets)] if min_ng is not None else [0] * n_datasets
        )

        min_bg_df = getattr(self.cfg, 'min_bg_df', 0)
        _max_retries = 20

        if getattr(self.cfg, 'loop', False) or self.cfg.ds_type in [
            'scm',
            'mixed',
        ]:  # Option A: loop
            datasets = []
            for i in tqdm(range(n_datasets), desc=desc):
                # If min_bg_df is set, retry until m − d ≥ min_bg_df.  Trimming X
                # while keeping Y intact creates mismatched ground truth (dropped
                # predictors inflate sigma_eps), so we redraw instead.
                retry_seeds = seedseqs[i].spawn(_max_retries)
                for attempt in range(_max_retries + 1):
                    seed = seedseqs[i] if attempt == 0 else retry_seeds[attempt - 1]
                    dataset = self._genDataset(
                        self.cfg,
                        seed,
                        d[i],
                        q[i],
                        ns_slices[i],
                        min_n_eff=min_n_effs[i],
                    )
                    m_i = int(dataset['ns'].shape[0])
                    d_i = int(dataset['d'])
                    if min_bg_df == 0 or m_i - d_i >= min_bg_df:
                        break
                datasets.append(dataset)
        else:  # Option B: parallelize
            datasets = Parallel(n_jobs=-1, backend='loky', batch_size='auto')(
                delayed(Generator._genDataset)(
                    self.cfg,
                    seedseqs[i],
                    d[i],
                    q[i],
                    ns_slices[i],
                    min_n_effs[i],
                )
                for i in tqdm(range(n_datasets), desc=desc)
            )
            # joblib returns list[Any], so for type safety we cast it
            datasets = cast(list[dict[str, np.ndarray]], datasets)

        # Enforce min_bg_df for q (trimming q is safe: rfx are zero-mean latents
        # that don't create Y-X mismatch the way dropping X columns would).
        if min_bg_df > 0:
            datasets = [Generator._clampQ(ds, min_bg_df) for ds in datasets]

        return datasets

    @staticmethod
    def _clampQ(ds: dict[str, np.ndarray], min_bg_df: int) -> dict[str, np.ndarray]:
        """Reduce q so that q(q+1)/2 + min_bg_df ≤ m (Ψ identifiability).

        Handles cases where the Emulator returned fewer groups than requested,
        causing m to fall below the identifiability floor for the current q.
        Trimming q is safe because rfx are zero-mean latents: dropping rfx
        columns does not create a Y-X mismatch the way dropping X columns would
        (the residuals absorb the dropped rfx variance symmetrically, without
        systematically inflating sigma_eps).  d is left unchanged since
        fixed-effect estimation pools over all n observations, not group means.

        The largest feasible q satisfies q(q+1)/2 ≤ m − min_bg_df, solved via
        the positive root of the quadratic: q* = floor((√(1+8k)−1)/2), k = m−min_bg_df.
        """
        m = int(ds['ns'].shape[0])
        q = int(ds['q'])
        k = max(0, m - min_bg_df)
        q_max_feasible = max(1, int(np.floor((np.sqrt(1 + 8 * k) - 1) / 2)))
        if q_max_feasible >= q:
            return ds
        q_new = q_max_feasible
        ds = dict(ds)
        ds['q'] = np.array(q_new)
        ds['rfx'] = ds['rfx'][:, :q_new]
        ds['sigma_rfx'] = ds['sigma_rfx'][:q_new]
        ds['corr_rfx'] = ds['corr_rfx'][:q_new, :q_new]
        return ds

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

    def saveConfig(self):
        """Save generation config to dataset directory for reproducibility."""
        dataset_dir = self.outdir / self.cfg.data_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save full resolved config (excluding runtime-only params)
        cfg_dict = vars(self.cfg).copy()
        # Remove runtime-only values that shouldn't be persisted
        for key in ['partition', 'begin', 'epochs', 'loop']:
            cfg_dict.pop(key, None)

        config_path = dataset_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(cfg_dict, f, sort_keys=False, default_flow_style=False)
        logger.info(f'Saved config to {config_path}')

    def genTest(self):
        logger.info('Generating test set...')
        ds_test = self._genBatch(n_datasets=self.cfg.bs_test, mini_batch_size=1)
        ds_test = aggregate(ds_test)
        ds_test = self._castCompactTypes(ds_test)
        dataset_dir = self.outdir / self.cfg.data_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        fn = dataset_dir / datasetFilename('test')
        np.savez_compressed(fn, **ds_test, allow_pickle=True)
        logger.info(f'Saved test set to {fn}')

    def genValid(self):
        logger.info('Generating validation set...')
        ds_valid = self._genBatch(n_datasets=self.cfg.bs_valid, mini_batch_size=1)
        ds_valid = aggregate(ds_valid)
        ds_valid = self._castCompactTypes(ds_valid)
        dataset_dir = self.outdir / self.cfg.data_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        fn = dataset_dir / datasetFilename('valid')
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
        dataset_dir = self.outdir / self.cfg.data_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.cfg.begin, self.cfg.epochs + 1):
            ds_train = self._genBatch(
                n_datasets=self.cfg.bs_train,
                mini_batch_size=self.cfg.bs_mini,
                epoch=epoch,
            )
            ds_train = aggregate(ds_train)
            ds_train = self._castCompactTypes(ds_train)
            fn = dataset_dir / datasetFilename('train', epoch)
            np.savez_compressed(fn, **ds_train, allow_pickle=True)
            logger.info(f'Saved training set to {fn}')

    @property
    def info(self) -> str:
        cfg = self.cfg
        epoch_range = f'{cfg.begin}–{cfg.epochs}' if cfg.partition in ['train', 'all'] else 'n/a'
        return f"""
====================
data id:    {cfg.data_id}
likelihood: {LIKELIHOOD_FAMILIES[cfg.likelihood_family]}
ds type:    {cfg.ds_type}
partition:  {cfg.partition}
epochs:     {epoch_range}
d:          1–{cfg.max_d}
q:          1–{cfg.max_q}
m:          {cfg.min_m}–{self.max_m_feasible}
n:          {cfg.min_n}–{cfg.max_n}
n_total:    ≤{cfg.max_n_total}
bs_train:   {cfg.bs_train}
bs_valid:   {cfg.bs_valid}
bs_test:    {cfg.bs_test}
===================="""

    def go(self):
        # Save config before generation
        self.saveConfig()

        if self.cfg.partition == 'test':
            self.genTest()
        elif self.cfg.partition == 'valid':
            self.genValid()
        elif self.cfg.partition == 'train':
            self.genTrain()
        elif self.cfg.partition == 'eval':
            self.cfg.partition = 'valid'
            self.genValid()
            self.cfg.partition = 'test'
            self.genTest()
        elif self.cfg.partition == 'all':
            self.cfg.partition = 'valid'
            self.genValid()
            self.cfg.partition = 'test'
            self.genTest()
            self.cfg.partition = 'train'
            self.genTrain()
        else:
            raise NotImplementedError(
                f'the partition type must be in [train, valid, test], but is {self.cfg.partition}'
            )


# -----------------------------------------------------------------------------
def main() -> None:
    cfg = setup()
    generator = Generator(cfg)
    print(generator.info)
    generator.go()


if __name__ == '__main__':
    main()
