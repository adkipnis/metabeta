import time
import sys
import gc
import resource
import logging
import argparse
import re
import zipfile
from dataclasses import replace
from itertools import chain
from pathlib import Path

import matplotlib
import numpy as np
from numpy.lib import format as npFormat
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.utils.logger import setupLogging
from metabeta.utils.device import setDevice, synchronizeDevice
from metabeta.utils.names import datasetFilename, runName
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import (
    modelFromYaml,
    ApproximatorConfig,
    assimilateConfig,
    loadDataConfig,
)
from metabeta.utils.templates import loadConfigFromCheckpoint
from metabeta.utils.dataloader import Dataloader, toDevice, subsetBatch
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import (
    EvaluationSummary,
    dictMean,
    nutsConvergeMask,
    subsetProposal,
)
from metabeta.utils.results import Proposal, concatProposalsBatch
from metabeta.models.approximator import Approximator
from metabeta.utils.posterior_cache import (
    cacheRefMtime,
    loadProposalCache,
    maskTag,
    posteriorSampleCacheName,
    saveProposalCache,
)
from metabeta.utils.posterior_eval import (
    IMH_METHODS,
    fit2proposal,
    fitBatchMask,
    loadOrRefine,
    posthocDefaults,
    validMethods,
)
from metabeta.evaluation.intervals import getCoverageErrors, getCoverages, getCredibleIntervals
from metabeta.evaluation.point import getCorrelation, getPointEstimates, getRMSE
from metabeta.evaluation.summary import EST_TYPE, _averageOverAlpha, getSummary, summaryTable
from metabeta.plotting import plotComparison

logger = logging.getLogger('evaluate.py')

_ALL_MODELS = ('MB', 'NUTS', 'ADVI', 'LAPLACE')
_FIT_MODELS = frozenset(('NUTS', 'ADVI', 'LAPLACE'))

# Post-hoc IMH refinement layered on the raw MB flow posterior. Not part of "all": it costs a
# refinement pass on top of MB sampling, so it is opt-in via --models MB+IMH.
_MB_IMH = 'MB+IMH'
_KNOWN_MODELS = _ALL_MODELS + (_MB_IMH,)
# Models that come from the checkpoint rather than from cached fits in the *.fit.npz.
_MB_MODELS = frozenset(('MB', _MB_IMH))
# Summary-cache methods keyed only by the data file (as opposed to checkpoint-derived ones).
_FIT_METHODS = frozenset(m.lower() for m in _FIT_MODELS)

# Cache filenames used to carry the pseudo-MoE view count. That path is gone, but the tag
# stays in the key so caches written before its removal (all k=0) still resolve.
_LEGACY_K_TAG = 'k0'


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # Primary: load from checkpoint (required when MB is evaluated)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--prefix', type=str, default='latest', help='Checkpoint prefix: best or latest')
    # Data-direct: evaluate fit-only models without a checkpoint
    parser.add_argument('--data_path_test', type=str, help='Direct path to test.fit.npz (no checkpoint needed for fit models)')
    parser.add_argument('--data_path_valid', type=str, help='Direct path to valid.fit.npz')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--r_tag', type=str)
    parser.add_argument('--data_id', type=str)
    parser.add_argument('--data_id_valid', type=str)
    parser.add_argument('--data_id_test', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_tables', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--outdir', type=str)
    parser.add_argument(
        '--partition', type=str, default='test', choices=['valid', 'test', 'all'],
        help='Data partition(s) to evaluate: valid, test, or all (default: test)',
    )
    parser.add_argument(
        '--models', type=str, default='all',
        help='Models to evaluate: comma-separated MB/MB+IMH/NUTS/ADVI/LAPLACE or "all" '
             '(default: all; "all" excludes MB+IMH)',
    )
    parser.add_argument(
        '--imh_method', type=str, default=None, choices=list(IMH_METHODS),
        help='IMH variant behind MB+IMH (default: the per-family default in presets.yaml)',
    )
    parser.add_argument(
        '--converged_subset', action=argparse.BooleanOptionalAction, default=False,
        help='Also evaluate on the NUTS-converged subset',
    )
    parser.add_argument(
        '--convergence_mode', type=str, default='liberal', choices=['strict', 'liberal'],
        help='NUTS convergence filter mode (default: liberal)',
    )
    parser.add_argument(
        '--pareto_k_thr', type=float, default=0.7,
        help='Pareto-k threshold for LOO-NLL subset (default: 0.7)',
    )
    parser.add_argument(
        '--pred_coverage', action=argparse.BooleanOptionalAction, default=False,
        help='Compute predictive interval coverage/width',
    )
    parser.add_argument(
        '--summary_chunk_size', type=int, default=4,
        help='Datasets per predictive-summary chunk (default: 4)',
    )
    parser.add_argument(
        '--plot', action=argparse.BooleanOptionalAction, default=True,
        help='Save comparison plots (default: true)',
    )
    parser.add_argument(
        '--show', action=argparse.BooleanOptionalAction, default=False,
        help='Open the figure in an interactive window; blocks until closed (default: false)',
    )
    parser.add_argument(
        '--warmup', action=argparse.BooleanOptionalAction, default=True,
        help='Run an untimed one-sample MB warm-up before timing model evaluation (default: true)',
    )
    parser.add_argument(
        '--comparison_legend', type=str, choices=['panel', 'right'], default='right',
        help='Comparison plot legend placement (default: right)',
    )
    parser.add_argument(
        '--plot_suffix', type=str, default='',
        help='Optional suffix for comparison plot filenames, e.g. "with_laplace"',
    )
    # fmt: on
    args = parser.parse_args()

    if hasattr(args, 'checkpoint') and args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        cfg_dict = loadConfigFromCheckpoint(checkpoint_path)
        cfg_dict['_checkpoint_dir'] = str(checkpoint_path)
        cfg_dict['_checkpoint_prefix'] = args.prefix
        for k, v in vars(args).items():
            if v is not None and k not in ['checkpoint', 'prefix', 'name']:
                cfg_dict[k] = v
    elif (hasattr(args, 'data_path_test') and args.data_path_test) or (
        hasattr(args, 'data_path_valid') and args.data_path_valid
    ):
        cfg_dict = {k: v for k, v in vars(args).items() if v is not None}
        models_str = cfg_dict.get('models', 'all')
        active = [
            x.strip().upper()
            for x in (list(_ALL_MODELS) if models_str == 'all' else models_str.split(','))
        ]
        if any(m in _MB_MODELS for m in active):
            raise ValueError(
                '--data_path_test/--data_path_valid mode: use fit models only '
                '(no MB / MB+IMH without --checkpoint)'
            )
    else:
        raise ValueError(
            'Must specify one of:\n'
            '  1. Checkpoint (required for MB): --checkpoint <dir> [--prefix best|latest]\n'
            '  2. Data paths (NUTS/ADVI only): --data_path_test <path> [--data_path_valid <path>]\n'
        )

    if cfg_dict.get('save_tables') is None:
        cfg_dict['save_tables'] = True

    return argparse.Namespace(**cfg_dict)


# =============================================================================
class Evaluator:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.dir = Path(__file__).resolve().parent
        setSeed(cfg.seed)
        self.device = setDevice(cfg.device)

        self.cfg.batch_size = getattr(cfg, 'batch_size', 8)
        self.cfg.save_tables = getattr(cfg, 'save_tables', False)
        self.cfg.converged_subset = getattr(cfg, 'converged_subset', False)
        self.cfg.convergence_mode = getattr(cfg, 'convergence_mode', 'liberal')
        self.cfg.pareto_k_thr = getattr(cfg, 'pareto_k_thr', 0.7)
        self.cfg.plot = getattr(cfg, 'plot', True)
        self.cfg.show = getattr(cfg, 'show', False)
        self.cfg.warmup = getattr(cfg, 'warmup', True)
        self.cfg.plot_suffix = getattr(cfg, 'plot_suffix', '')
        self.cfg.summary_chunk_size = getattr(cfg, 'summary_chunk_size', 16)
        self.cfg.outdir = getattr(cfg, 'outdir', str(Path(self.dir, '..', 'outputs', 'results')))

        if hasattr(cfg, 'data_path_test') or hasattr(cfg, 'data_path_valid'):
            # data-direct mode: run name derived from the data directory
            data_p = Path(getattr(cfg, 'data_path_test', None) or cfg.data_path_valid)
            self.run_name = data_p.parent.name
            self.legacy_run_name = self.run_name
            self.ckpt_dir = None
        else:
            if hasattr(cfg, '_checkpoint_dir'):
                self.ckpt_dir = Path(cfg._checkpoint_dir)
                self.run_name = self.ckpt_dir.name
            else:
                self.run_name = runName(vars(cfg))
                self.ckpt_dir = Path(self.dir, '..', 'outputs', 'checkpoints', self.run_name)
            self.legacy_run_name = runName(vars(cfg))
        self.checkpoint_prefix = getattr(
            cfg,
            '_checkpoint_prefix',
            getattr(cfg, 'prefix', 'latest'),
        )

        self._initData()
        self.model_loaded = False
        self._imh_method_cache: str | None = None

        self.plot_dir = Path(self.dir, '..', 'outputs', 'plots', self.run_name)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = None
        if self.cfg.save_tables:
            base_dir = Path(self.cfg.outdir)
            self.results_dir = base_dir / self.run_name
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def _initData(self) -> None:
        if hasattr(self.cfg, 'data_path_test') or hasattr(self.cfg, 'data_path_valid'):
            self._initDataDirect()
        else:
            self._initDataFromConfig()

    @staticmethod
    def _maxRssMb() -> float:
        """Maximum resident set size reported by the OS, in MiB."""
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        return rss / (1024 * 1024) if sys.platform == 'darwin' else rss / 1024

    def _logMemory(self, msg: str, *args) -> None:
        logger.info('%s [max RSS %.1f MiB]', msg % args if args else msg, self._maxRssMb())

    def _tableOnlyMode(self) -> bool:
        return self.cfg.save_tables and not self.cfg.plot and not self.cfg.converged_subset

    def _directDataMode(self) -> bool:
        return hasattr(self.cfg, 'data_path_test') or hasattr(self.cfg, 'data_path_valid')

    def _partitionDataPath(self, partition: str) -> Path:
        return self.data_path_test if partition == 'test' else self.data_path_valid

    def _initDataFromConfig(self) -> None:
        self.data_cfg_train = loadDataConfig(self.cfg.data_id)
        assimilateConfig(self.cfg, self.data_cfg_train)
        self.cfg.data_id_valid = getattr(self.cfg, 'data_id_valid', self.cfg.data_id)
        self.cfg.data_id_test = getattr(self.cfg, 'data_id_test', self.cfg.data_id_valid)
        self.data_cfg_valid = loadDataConfig(self.cfg.data_id_valid)
        self.data_cfg_test = loadDataConfig(self.cfg.data_id_test)
        self.data_cfg = self.data_cfg_train
        self.data_path_valid = self._getDataPath('valid')
        self.data_path_test = self._getDataPath('test')
        self.dl_valid = None
        self.dl_test = None
        logger.info('Dataloaders will be constructed lazily for requested partitions only.')

    def _initDataDirect(self) -> None:
        """Initialise data from explicit file paths; infers config fields from the npz."""
        test_p = Path(self.cfg.data_path_test) if hasattr(self.cfg, 'data_path_test') else None
        valid_p = Path(self.cfg.data_path_valid) if hasattr(self.cfg, 'data_path_valid') else None
        self.data_path_test = test_p or valid_p
        self.data_path_valid = valid_p or test_p
        self.dl_test = None
        self.dl_valid = None
        self._inferConfigFromNpz(self.data_path_test)
        self.data_cfg = {}
        logger.info('Dataloaders will be constructed lazily for requested partitions only.')

    def _inferConfigFromNpz(self, path: Path) -> None:
        with np.load(path, allow_pickle=True) as raw:
            if not hasattr(self.cfg, 'max_d'):
                self.cfg.max_d = int(raw['d'].max())
            if not hasattr(self.cfg, 'max_q'):
                self.cfg.max_q = int(raw['q'].max())
            if not hasattr(self.cfg, 'likelihood_family'):
                raw_lf = raw['likelihood_family'] if 'likelihood_family' in raw.files else None
                self.cfg.likelihood_family = int(raw_lf[0]) if raw_lf is not None else 0
            if not hasattr(self.cfg, 'rescale'):
                self.cfg.rescale = False

    def _getDataPath(self, partition: str, prefer_fit: bool = True) -> Path:
        data_cfg = self.data_cfg_test if partition == 'test' else self.data_cfg_valid
        data_fname = datasetFilename(partition)
        data_subdir = data_cfg['data_id']
        data_path = Path(self.dir, '..', 'outputs', 'data', data_subdir, data_fname)
        fit_path = data_path.with_suffix('.fit.npz')
        if prefer_fit and fit_path.exists():
            data_path = fit_path
        assert data_path.exists(), f'data file not found: {data_path}'
        return data_path

    def _getDataLoader(
        self,
        partition: str,
        batch_size: int | None = None,
        prefer_fit: bool = True,
        sortish: bool | None = None,
    ) -> tuple[Dataloader, Path]:
        if hasattr(self.cfg, 'data_path_test') or hasattr(self.cfg, 'data_path_valid'):
            data_path = self._partitionDataPath(partition)
        else:
            data_path = self._getDataPath(partition, prefer_fit=prefer_fit)
        if sortish is None:
            sortish = batch_size is not None
        self._logMemory(
            'Loading %s dataloader from %s; this loads the full npz collection',
            partition,
            data_path,
        )
        dl = Dataloader(
            data_path,
            batch_size=batch_size,
            sortish=sortish,
            max_d=getattr(self.cfg, 'max_d', None),
            max_q=getattr(self.cfg, 'max_q', None),
        )
        return dl, data_path

    def _baseDataLoader(self, partition: str) -> tuple[Dataloader, Path]:
        return self._getDataLoader(
            partition,
            batch_size=self.cfg.batch_size,
            prefer_fit=False,
            sortish=False,
        )

    def _fitDataPath(self, partition: str) -> Path:
        data_path = self._partitionDataPath(partition)
        if data_path.name.endswith('.fit.npz'):
            return data_path
        fit_path = data_path.with_suffix('.fit.npz')
        assert fit_path.exists(), f'fit data file not found: {fit_path}'
        return fit_path

    def _ensureDataloader(self, partition: str) -> None:
        if partition == 'valid' and self.dl_valid is None:
            self.dl_valid, self.data_path_valid = self._getDataLoader(
                'valid', batch_size=self.cfg.batch_size
            )
        elif partition == 'test' and self.dl_test is None:
            self.dl_test, self.data_path_test = self._getDataLoader(
                'test', batch_size=self.cfg.batch_size
            )

    def _setDataloader(
        self,
        partition: str,
        dl: Dataloader,
        path: Path,
    ) -> None:
        if partition == 'test':
            self.dl_test = dl
            self.data_path_test = path
        else:
            self.dl_valid = dl
            self.data_path_valid = path

    def _ensureFitDataloader(self, partition: str) -> None:
        current = self.dl_test if partition == 'test' else self.dl_valid
        if current is not None and not getattr(current, '_sortish', True):
            return
        dl, path = self._getDataLoader(
            partition,
            batch_size=self.cfg.batch_size,
            sortish=False,
        )
        self._setDataloader(partition, dl, path)

    def _initModel(self) -> None:
        if hasattr(self.cfg, 'model_cfg') and isinstance(self.cfg.model_cfg, ApproximatorConfig):
            self.model_cfg = self.cfg.model_cfg
        else:
            model_cfg_path = Path(self.dir, '..', 'configs', 'models', f'{self.cfg.model_id}.yaml')
            self.model_cfg = modelFromYaml(
                model_cfg_path,
                d_ffx=self.cfg.max_d,
                d_rfx=self.cfg.max_q,
                likelihood_family=self.cfg.likelihood_family,
            )
        self.model = Approximator(self.model_cfg).to(self.device)
        self.model.eval()

    def _load(self) -> None:
        prefix = getattr(self, 'checkpoint_prefix', getattr(self.cfg, 'prefix', 'best'))
        path = Path(self.ckpt_dir, prefix + '.pt')
        assert path.exists(), f'checkpoint not found: {path}'
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if self.data_cfg != payload['data_cfg']:
            logger.warning('data config mismatch between current and checkpoint')
        if self.model_cfg.to_dict() != payload['model_cfg']:
            logger.warning('model config mismatch between current and checkpoint')
        self.model.load_state_dict(payload['model_state'])
        if self.cfg.compile and self.device.type == 'cuda':
            self.model.compile()

    def _ensureModelLoaded(self) -> None:
        if self.model_loaded:
            return
        self._initModel()
        self._load()
        self.model_loaded = True

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def _fit2proposal(self, batch: dict[str, torch.Tensor], prefix: str) -> Proposal:
        """Shared fit -> Proposal conversion, plus this pipeline's optional rescale."""
        proposal = fit2proposal(batch, prefix)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        return proposal

    @staticmethod
    def _npzArray(
        raw: np.lib.npyio.NpzFile,
        key: str,
        mask: np.ndarray | None = None,
        dtype=np.float32,
    ) -> np.ndarray:
        arr = raw[key]
        if mask is not None:
            arr = arr[mask]
        return np.asarray(arr, dtype=dtype)

    @staticmethod
    def _streamNpzArray(
        fit_path: Path,
        key: str,
        mask: np.ndarray | None = None,
        dtype=np.float32,
        axes: tuple[int, ...] | None = None,
        chunk_bytes: int = 256 << 20,
    ) -> np.ndarray:
        """Read ``key`` from a compressed npz in dataset blocks, without materialising it whole.

        ``np.load(...)[key]`` decompresses the full stored array (float64 for the fit samples)
        before any downcast or mask, which for ``*_rfx`` is several GB of pure transient. Here
        the zip entry is consumed sequentially along axis 0 (datasets), each block is cast to
        ``dtype`` and, when ``axes`` is given, transposed straight into its final layout — so
        peak usage is the result plus one block rather than result + source + permute copy.

        ``axes`` is the permutation of the *stored* axes, as passed to ``np.transpose``; axis 0
        must stay first, since that is the axis being streamed.
        """
        if axes is not None and axes[0] != 0:
            raise ValueError(f'axes must keep the dataset axis first, got {axes}')
        with zipfile.ZipFile(fit_path) as archive:
            with archive.open(f'{key}.npy') as handle:
                version = npFormat.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, src_dtype = npFormat.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran_order, src_dtype = npFormat.read_array_header_2_0(handle)
                else:
                    raise ValueError(f'unsupported npy version {version} for {key}')
                if fortran_order:
                    raise NotImplementedError(
                        f'{key} is Fortran-ordered; streaming assumes C order'
                    )

                n_datasets = shape[0]
                keep = np.ones(n_datasets, dtype=bool) if mask is None else np.asarray(mask, bool)
                row_shape = shape[1:]
                row_bytes = (
                    int(np.prod(row_shape)) * src_dtype.itemsize
                    if row_shape
                    else (src_dtype.itemsize)
                )
                out_row_shape = row_shape if axes is None else tuple(shape[a] for a in axes)[1:]
                out = np.empty((int(keep.sum()),) + out_row_shape, dtype=dtype)

                rows_per_chunk = max(1, chunk_bytes // max(row_bytes, 1))
                written = 0
                for start in range(0, n_datasets, rows_per_chunk):
                    stop = min(start + rows_per_chunk, n_datasets)
                    buffer = handle.read((stop - start) * row_bytes)
                    block = np.frombuffer(buffer, dtype=src_dtype).reshape(
                        (stop - start,) + row_shape
                    )
                    selected = keep[start:stop]
                    if not selected.all():
                        block = block[selected]
                    if block.shape[0] == 0:
                        continue
                    if axes is not None:
                        block = block.transpose(axes)
                    out[written : written + block.shape[0]] = block
                    written += block.shape[0]
        return out

    def _fitProposalFromNpz(
        self,
        fit_path: Path,
        method: str,
        mask: np.ndarray | None = None,
        scale: torch.Tensor | None = None,
    ) -> Proposal:
        prefix = method.lower()
        self._logMemory('Loading %s samples directly from %s', prefix, fit_path)
        with np.load(fit_path, allow_pickle=True) as raw:
            ffx = torch.as_tensor(self._npzArray(raw, f'{prefix}_ffx', mask)).permute(0, 2, 1)
            sigma_rfx = torch.as_tensor(self._npzArray(raw, f'{prefix}_sigma_rfx', mask)).permute(
                0, 2, 1
            )
            samples_g = [ffx.contiguous(), sigma_rfx.contiguous()]

            has_sigma_eps = f'{prefix}_sigma_eps' in raw.files
            if has_sigma_eps:
                sigma_eps = self._npzArray(raw, f'{prefix}_sigma_eps', mask)
                sigma_eps = np.squeeze(sigma_eps, axis=1) if sigma_eps.ndim == 3 else sigma_eps
                samples_g.append(torch.as_tensor(sigma_eps).unsqueeze(-1).contiguous())

            corr_rfx = None
            if f'{prefix}_corr_rfx' in raw.files:
                corr = self._npzArray(raw, f'{prefix}_corr_rfx', mask)
                corr = np.squeeze(corr, axis=1) if corr.ndim == 5 and corr.shape[1] == 1 else corr
                corr_rfx = torch.as_tensor(corr).contiguous()

            duration = self._npzArray(raw, f'{prefix}_duration', mask, dtype=np.float64)

        # (b, q, m, s) on disk -> (b, m, s, q); streamed because this array dominates the load
        rfx = torch.from_numpy(
            self._streamNpzArray(fit_path, f'{prefix}_rfx', mask=mask, axes=(0, 2, 3, 1))
        )

        proposed = {
            'global': {'samples': torch.cat(samples_g, dim=-1)},
            'local': {'samples': rfx},
        }
        proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps, corr_rfx=corr_rfx)
        proposal.tpd = float(np.nanmean(duration))
        if self.cfg.rescale:
            if scale is None:
                raise ValueError('scale is required to rescale fit proposals loaded from npz')
            proposal.rescale(scale)
        self._logMemory('Loaded %s proposal directly from npz', prefix)
        return proposal

    def _sampleBatch(
        self, batch: dict[str, torch.Tensor], n_samples: int | None = None
    ) -> Proposal:
        proposal = self.model.estimate(batch, n_samples=n_samples or self.cfg.n_samples)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        return proposal

    def _warmupMbBatch(self, batch: dict[str, torch.Tensor], label: str) -> None:
        if not self.cfg.warmup:
            return

        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        synchronizeDevice(self.device)
        logger.info('Warming MB model on one %s batch with n_samples=1', label)
        try:
            proposal = self._sampleBatch(batch, n_samples=1)
            del proposal
            synchronizeDevice(self.device)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    @torch.no_grad()
    def sampleMinibatched(self, dl: Dataloader, label: str) -> Proposal:
        self._ensureModelLoaded()
        proposals = []
        n_datasets = 0
        iterator = iter(dl)
        try:
            first_batch = toDevice(next(iterator), self.device)
        except StopIteration:
            raise ValueError(f'cannot sample empty dataloader for {label}')
        self._warmupMbBatch(first_batch, label)
        synchronizeDevice(self.device)
        t0 = time.perf_counter()
        for batch in tqdm(chain([first_batch], iterator), total=len(dl), desc=f'  {label}'):
            if batch is not first_batch:
                batch = toDevice(batch, self.device)
            proposal = self._sampleBatch(batch)
            proposal.to('cpu')
            proposals.append(proposal)
            n_datasets += batch['X'].shape[0]
        synchronizeDevice(self.device)
        t1 = time.perf_counter()
        merged = concatProposalsBatch(proposals)
        merged.tpd = (t1 - t0) / max(n_datasets, 1)
        return merged

    # -------------------------------------------------------------------------
    # Summary helpers
    # -------------------------------------------------------------------------

    def summary(
        self,
        proposal: Proposal,
        batch: dict[str, torch.Tensor],
    ) -> EvaluationSummary:
        batch = toDevice(batch, 'cpu')
        if self.cfg.rescale:
            batch = rescaleData(batch)
        proposal.to('cpu')
        lf = self.cfg.likelihood_family
        pred_cov = getattr(self.cfg, 'pred_coverage', False)
        eval_summary = getSummary(
            proposal,
            batch,
            likelihood_family=lf,
            compute_pred_coverage=pred_cov,
            dataset_chunk_size=self.cfg.summary_chunk_size,
        )
        logger.info(summaryTable(eval_summary, lf))
        return eval_summary

    def _summaryCachePath(
        self,
        partition: str,
        method: str,
        mask: np.ndarray | None = None,
        run_name: str | None = None,
        prefix: str | None = None,
    ) -> Path:
        data_path = self._partitionDataPath(partition)
        if method not in _FIT_METHODS:
            # checkpoint-derived (MB and its post-hoc refinements): key by run/prefix/sampling
            run_name = run_name or self.run_name
            prefix = prefix or getattr(
                self,
                'checkpoint_prefix',
                getattr(self.cfg, 'prefix', 'best'),
            )
            cache_name = (
                f'summary_{partition}_{method}_{run_name}_{prefix}'
                f'_s{self.cfg.n_samples}_seed{self.cfg.seed}_{_LEGACY_K_TAG}'
                f'_predcov{int(getattr(self.cfg, "pred_coverage", False))}'
                f'_{maskTag(mask)}.pt'
            )
            return data_path.parent / cache_name
        if mask is not None:
            return data_path.parent / f'summary_{partition}_{method}_{maskTag(mask)}.pt'
        return data_path.parent / f'summary_{partition}_{method}.pt'

    def _mbSampleCachePath(
        self,
        partition: str,
        run_name: str | None = None,
        prefix: str | None = None,
    ) -> Path:
        data_path = self._partitionDataPath(partition)
        run_name = run_name or self.run_name
        prefix = prefix or getattr(
            self,
            'checkpoint_prefix',
            getattr(self.cfg, 'prefix', 'best'),
        )
        cache_name = posteriorSampleCacheName(
            partition=partition,
            method='mb',
            checkpoint_name=run_name,
            checkpoint_prefix=prefix,
            n_samples=self.cfg.n_samples,
            seed=self.cfg.seed,
        )
        return data_path.parent / cache_name

    def _mbSampleCacheCandidates(self, partition: str) -> list[Path]:
        paths = [self._mbSampleCachePath(partition)]
        legacy_run_name = getattr(self, 'legacy_run_name', self.run_name)
        if legacy_run_name != self.run_name:
            for prefix in (self.checkpoint_prefix, 'latest'):
                paths.append(
                    self._mbSampleCachePath(partition, run_name=legacy_run_name, prefix=prefix)
                )
        return list(dict.fromkeys(paths))

    def _mbSampleRefMtime(self, partition: str) -> float:
        data_path = (
            self._getDataPath(partition, prefer_fit=False)
            if not self._directDataMode()
            else self._partitionDataPath(partition)
        )
        return cacheRefMtime(data_path, getattr(self, 'ckpt_dir', None), self.checkpoint_prefix)

    def _saveMbSampleCache(self, partition: str, proposal: Proposal) -> Path:
        cache_path = self._mbSampleCachePath(partition)
        saveProposalCache(
            cache_path,
            proposal,
            metadata={
                'n_samples': self.cfg.n_samples,
                'seed': self.cfg.seed,
                'checkpoint_prefix': self.checkpoint_prefix,
                'run_name': self.run_name,
            },
        )
        logger.info('Saved MB posterior samples to %s', cache_path)
        return cache_path

    def _loadMbSampleCache(self, partition: str) -> Proposal | None:
        ref_mtime = self._mbSampleRefMtime(partition)
        preferred = self._mbSampleCachePath(partition)
        stale: list[Path] = []
        for cache_path in self._mbSampleCacheCandidates(partition):
            if not cache_path.exists():
                continue
            if cache_path.stat().st_mtime < ref_mtime:
                stale.append(cache_path)
                continue
            try:
                proposal, _ = loadProposalCache(cache_path)
            except (KeyError, ValueError) as exc:
                logger.warning('Ignoring invalid MB sample cache %s: %s', cache_path, exc)
                continue
            if cache_path != preferred and not preferred.exists():
                self._saveMbSampleCache(partition, proposal)
                logger.info('Copied cached MB posterior samples to %s', preferred)
            logger.info('Loading cached MB posterior samples from %s', cache_path)
            return proposal
        if stale:
            logger.info(
                'Cached MB posterior samples exist but are older than data/checkpoint reference: %s',
                '; '.join(str(path) for path in stale),
            )
        return None

    def _loadOrSampleMb(self, partition: str, dl: Dataloader) -> Proposal:
        cached = self._loadMbSampleCache(partition)
        if cached is not None:
            return cached
        proposal = self.sampleMinibatched(dl, f'MB ({partition})')
        self._saveMbSampleCache(partition, proposal)
        return proposal

    def _summaryRefMtime(self, partition: str, method: str) -> float:
        # fit summaries invalidate on the data alone; checkpoint-derived ones also on the model
        ckpt_dir = None if method in _FIT_METHODS else getattr(self, 'ckpt_dir', None)
        return cacheRefMtime(self._partitionDataPath(partition), ckpt_dir, self.checkpoint_prefix)

    def _summaryCacheCandidates(
        self,
        partition: str,
        method: str,
        mask: np.ndarray | None = None,
    ) -> list[Path]:
        paths = [self._summaryCachePath(partition, method, mask=mask)]
        legacy_run_name = getattr(self, 'legacy_run_name', self.run_name)
        if method not in _FIT_METHODS and legacy_run_name != self.run_name:
            for prefix in (self.checkpoint_prefix, 'latest'):
                paths.append(
                    self._summaryCachePath(
                        partition,
                        method,
                        mask=mask,
                        run_name=legacy_run_name,
                        prefix=prefix,
                    )
                )
        return list(dict.fromkeys(paths))

    def _loadCachedSummary(
        self,
        partition: str,
        method: str,
        mask: np.ndarray | None = None,
    ) -> EvaluationSummary | None:
        ref_mtime = self._summaryRefMtime(partition, method)
        preferred = self._summaryCachePath(partition, method, mask=mask)
        stale: list[Path] = []
        for cache_path in self._summaryCacheCandidates(partition, method, mask=mask):
            if cache_path.exists() and cache_path.stat().st_mtime >= ref_mtime:
                logger.info('Loading cached %s/%s summary from %s', partition, method, cache_path)
                try:
                    summary = EvaluationSummary.load(cache_path)
                except (KeyError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        'Ignoring invalid %s/%s summary cache %s: %s',
                        partition,
                        method,
                        cache_path,
                        exc,
                    )
                    continue
                if cache_path != preferred and not preferred.exists():
                    summary.save(preferred)
                    logger.info('Copied cached %s/%s summary to %s', partition, method, preferred)
                return summary
            if cache_path.exists():
                stale.append(cache_path)
        if stale:
            logger.info(
                'Cached %s/%s summary exists but is older than its data/checkpoint reference: %s',
                partition,
                method,
                '; '.join(str(path) for path in stale),
            )
        return None

    def _loadOrComputeSummary(
        self,
        proposal: Proposal,
        batch: dict[str, torch.Tensor],
        partition: str,
        method: str,
        mask: np.ndarray | None = None,
    ) -> EvaluationSummary:
        cache_path = self._summaryCachePath(partition, method, mask=mask)
        cached = self._loadCachedSummary(partition, method, mask=mask)
        if cached is not None:
            if cached.tpd is None and proposal.tpd is not None:
                cached.tpd = proposal.tpd
                cached.save(cache_path)
                logger.info('Updated cached %s/%s summary tpd in %s', partition, method, cache_path)
            return cached
        result = self.summary(proposal, batch)
        result.save(cache_path)
        logger.info('Saved %s/%s summary to %s', partition, method, cache_path)
        return result

    def _datasetCountFromPath(self, path: Path) -> int:
        with np.load(path, allow_pickle=True) as raw:
            return int(raw['y'].shape[0])

    def _fitMaskFromPath(self, path: Path, method: str) -> np.ndarray | None:
        failed_key = f'{method.lower()}_failed'
        with np.load(path, allow_pickle=True) as raw:
            if failed_key not in raw.files:
                return None
            mask = ~raw[failed_key].astype(bool)
            return None if mask.all() else mask

    def _durationMeanFromPath(
        self,
        path: Path,
        method: str,
        mask: np.ndarray | None = None,
    ) -> float | None:
        duration_key = f'{method.lower()}_duration'
        with np.load(path, allow_pickle=True) as raw:
            if duration_key not in raw.files:
                return None
            durations = np.asarray(raw[duration_key], dtype=np.float64).reshape(-1)
        if mask is not None:
            durations = durations[mask]
        durations = durations[np.isfinite(durations)]
        return float(durations.mean()) if durations.size else None

    def _imhMethod(self) -> str:
        """IMH variant behind MB+IMH: --imh_method, else the presets.yaml per-family default."""
        if getattr(self, '_imh_method_cache', None) is not None:
            return self._imh_method_cache
        lf = self.cfg.likelihood_family
        method = getattr(self.cfg, 'imh_method', None)
        if method is None:
            defaults = [m for m in posthocDefaults(lf) if m in IMH_METHODS]
            if not defaults:
                raise ValueError(
                    f'no IMH default in presets.yaml for likelihood_family={lf}; '
                    f'pass --imh_method explicitly (one of {IMH_METHODS})'
                )
            method = defaults[0]
        if not validMethods([method], lf):
            raise ValueError(f'{method} is incompatible with likelihood_family={lf}')
        self._imh_method_cache = method
        logger.info('MB+IMH refinement method: %s (likelihood_family=%d)', method, lf)
        return method

    def _methodName(self, model: str) -> str:
        """Summary-cache method key for a model (MB+IMH resolves to its IMH variant)."""
        return self._imhMethod() if model == _MB_IMH else model.lower()

    def _refineMb(
        self,
        partition: str,
        base: Proposal,
        full_batch: dict[str, torch.Tensor],
    ) -> Proposal:
        """Layer post-hoc IMH on the raw MB proposal (both in rescaled data space)."""
        method = self._imhMethod()
        # refineProposal expects proposal and batch in the same space; _loadOrSampleMb already
        # rescaled the proposal, so the batch has to follow (rescaleData copies, no mutation).
        refine_batch = rescaleData(full_batch) if self.cfg.rescale else full_batch
        proposal, refine_seconds = loadOrRefine(
            method,
            base,
            refine_batch,
            self._partitionDataPath(partition),
            self.ckpt_dir,
            self.checkpoint_prefix,
            self.cfg.n_samples,
            self.cfg.seed,
            self.cfg.likelihood_family,
            self.cfg.rescale,
            None,
            self.cfg.batch_size,
        )
        n = max(full_batch['X'].shape[0], 1)
        proposal.tpd = (base.tpd or 0.0) + refine_seconds / n
        return proposal

    def _activeModels(self, partition: str, models: list[str]) -> list[str]:
        has_fits = self._hasFits(partition)
        active = [m for m in models if m in _MB_MODELS or (has_fits and m in _FIT_MODELS)]
        if not active:
            logger.warning('No active models for partition=%s (no fit file found)', partition)
        return active

    @staticmethod
    def _fitSummaryMask(
        src_mask: np.ndarray | None,
        common_mask: np.ndarray | None,
        n: int,
    ) -> np.ndarray | None:
        comparison_mask = np.ones(n, dtype=bool) if common_mask is None else common_mask
        native_mask = np.ones(n, dtype=bool) if src_mask is None else src_mask
        return None if np.array_equal(native_mask, comparison_mask) else common_mask

    def _cachedRowsForPartition(
        self,
        partition: str,
        models: list[str],
        fit_label: str,
        multi: bool,
    ) -> list[dict] | None:
        """Return rows from existing summary caches without loading a Dataloader.

        This is intentionally conservative. If any required cache is missing, or if common-subset
        comparison would require a cache that does not exist, the caller should fall back to the
        normal evaluation path.
        """
        data_path = self._partitionDataPath(partition)
        active = self._activeModels(partition, models)
        if not active:
            return []

        n = self._datasetCountFromPath(data_path)
        masks = {
            model: self._fitMaskFromPath(data_path, model)
            for model in active
            if model in _FIT_MODELS
        }
        common_mask = self._commonMask(list(masks.values()), n)

        rows: list[dict] = []
        missing: list[str] = []
        for model in active:
            method = self._methodName(model)
            mask = common_mask
            if model in _FIT_MODELS:
                mask = self._fitSummaryMask(masks[model], common_mask, n)
            summary = self._loadCachedSummary(partition, method, mask=mask)
            if summary is None:
                missing.append(f'{model}({self._summaryCachePath(partition, method, mask=mask)})')
                continue
            if model in _FIT_MODELS and summary.tpd is None:
                duration = self._durationMeanFromPath(data_path, method, common_mask)
                if duration is not None:
                    summary.tpd = duration
                    summary.save(self._summaryCachePath(partition, method, mask=mask))
                    logger.info(
                        'Updated cached %s/%s summary tpd from %s_duration',
                        partition,
                        method,
                        method,
                    )
            label = self._displayLabel(model, partition, multi)
            rows.append(self._makeRow(label, summary, fit_label))

        if missing:
            logger.info(
                'Cached table fast path unavailable for partition=%s; unusable summaries: %s',
                partition,
                '; '.join(missing),
            )
            return None
        logger.info(
            'Loaded %d cached table rows for partition=%s without materializing full batches.',
            len(rows),
            partition,
        )
        return rows

    def _tableOnlyCacheError(self, partition: str) -> RuntimeError:
        data_path = self._partitionDataPath(partition)
        return RuntimeError(
            f'Table-only evaluation for partition={partition!r} cannot continue because at least '
            'one requested summary cache is missing or stale. Refusing to fall back to full '
            f'evaluation of {data_path}, which can materialize very large fit arrays. '
            'Recompute stale fit summaries with cache.py, rerun MB separately with --models MB, '
            'or run without --no-plot if full evaluation is intentional.'
        )

    def _canFallbackFromTableOnly(self, partition: str, models: list[str]) -> bool:
        active = self._activeModels(partition, models)
        return bool(active) and all(model in _MB_MODELS for model in active)

    def _canLightEvaluateFromTableOnly(self, partition: str, models: list[str]) -> bool:
        active = self._activeModels(partition, models)
        need_fits = any(model in _FIT_MODELS for model in active)
        return (
            bool(active)
            and need_fits
            and not self.cfg.converged_subset
            and not self._directDataMode()
        )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def plot(
        self,
        proposals: list[Proposal],
        summaries: list[EvaluationSummary],
        labels: list[str],
        batch: dict[str, torch.Tensor],
        plot_dir: Path | None = None,
    ) -> None:
        if self.cfg.rescale:
            batch = rescaleData(batch)
        summaries = [
            self._summaryForPlot(summary, proposal, batch)
            for summary, proposal in zip(summaries, proposals)
        ]
        target_dir = plot_dir if plot_dir is not None else self.plot_dir
        saved_path = plotComparison(
            summaries,
            proposals,
            labels,
            batch,
            plot_dir=target_dir,
            plot_name=self._plotName(self.cfg.plot_suffix),
            show=self.cfg.show,
            legend_right=getattr(self.cfg, 'comparison_legend', 'right') == 'right',
        )
        if saved_path is not None:
            logger.info('Saved comparison plot to %s', saved_path)

    def _fitLabel(self) -> str:
        return {0: 'ppR2', 1: 'ppAUC', 2: 'ppDev'}.get(self.cfg.likelihood_family, 'ppR2')

    @staticmethod
    def _displayModel(model: str) -> str:
        return 'LA' if model == 'LAPLACE' else model

    def _displayLabel(self, model: str, partition: str, multi: bool) -> str:
        label = self._displayModel(model)
        return f'{label}_{partition}' if multi else label

    @staticmethod
    def _plotName(suffix: str = '') -> str:
        suffix = suffix.strip().strip('_-')
        if not suffix:
            return 'comparison'
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', suffix)
        safe = safe.strip('._-')
        return f'comparison_{safe}' if safe else 'comparison'

    def _plotBatch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = (
            'X',
            'y',
            'ffx',
            'sigma_rfx',
            'sigma_eps',
            'corr_rfx',
            'rfx',
            'mask_d',
            'mask_q',
            'mask_mq',
            'sd_y',
        )
        return {key: batch[key] for key in keys if key in batch}

    @staticmethod
    def _summaryForPlot(
        summary: EvaluationSummary,
        proposal: Proposal,
        data: dict[str, torch.Tensor],
    ) -> EvaluationSummary:
        estimates = getPointEstimates(proposal, EST_TYPE)
        coverage = getCoverages(getCredibleIntervals(proposal), data)
        coverage_error = getCoverageErrors(coverage, log_ratio=False)
        log_coverage_ratio = getCoverageErrors(coverage, log_ratio=True)
        aggregated = replace(
            summary.aggregated,
            corr=getCorrelation(estimates, data),
            nrmse=getRMSE(estimates, data, normalize=True),
            coverage=coverage,
            ece=_averageOverAlpha(coverage_error),
            eace=_averageOverAlpha(coverage_error, absolute=True),
            lcr=_averageOverAlpha(log_coverage_ratio),
            abs_lcr=_averageOverAlpha(log_coverage_ratio, absolute=True),
            estimates=estimates,
        )
        return replace(summary, aggregated=aggregated)

    def _makeRow(self, label: str, summary: EvaluationSummary, fit_label: str) -> dict:
        ag, pd = summary.aggregated, summary.per_dataset
        return {
            'method': label,
            'R': dictMean(ag.corr),
            'NRMSE': dictMean(ag.nrmse),
            'ECE': dictMean(ag.ece),
            'EACE': dictMean(ag.eace),
            'RFX_joint_ECE': ag.rfx_joint_ece,
            'RFX_joint_EACE': ag.rfx_joint_eace,
            'LOO-NLL': pd.mloonll,
            'ppNLL': pd.mnll,
            fit_label: pd.mfit,
            'tpd': summary.tpd,
            'IS_eff': pd.meff,
            'Pareto_k': pd.mk,
            'ppEACE': pd.pp_eace,
            'ppWidth90': pd.pp_width_90,
        }

    def _bestIndices(
        self,
        rows: list[dict],
        metric_names: list[str],
        direction: dict[str, bool | str],
    ) -> dict[str, set[int]]:
        best: dict[str, set[int]] = {m: set() for m in metric_names}
        for metric in metric_names:
            values = [(i, r[metric]) for i, r in enumerate(rows) if r[metric] is not None]
            if not values:
                continue
            d = direction[metric]
            if d == 'abs':
                best_idx = min(values, key=lambda x: abs(x[1]))[0]
            elif d:
                best_idx = max(values, key=lambda x: x[1])[0]
            else:
                best_idx = min(values, key=lambda x: x[1])[0]
            best[metric].add(best_idx)
        return best

    def saveTables(self, rows: list[dict]) -> None:
        if self.results_dir is None:
            return
        metric_names = list(rows[0].keys())[1:]
        direction = {
            'R': True,
            'NRMSE': False,
            'ECE': 'abs',
            'EACE': False,
            'RFX_joint_ECE': 'abs',
            'RFX_joint_EACE': False,
            'LOO-NLL': False,
            'ppNLL': False,
            self._fitLabel(): True,
            'tpd': False,
            'IS_eff': True,
            'Pareto_k': False,
            'ppEACE': False,
            'ppWidth90': False,
        }
        direction = {k: v for k, v in direction.items() if k in metric_names}
        best = self._bestIndices(rows, metric_names, direction)

        def _fmt(val: float | None, i: int, metric: str, bold: tuple[str, str]) -> str:
            if val is None:
                return 'NA'
            cell = f'{val:.4f}'
            return f'{bold[0]}{cell}{bold[1]}' if i in best.get(metric, set()) else cell

        headers = ['Method'] + metric_names
        md_rows = [
            [r['method']] + [_fmt(r[m], i, m, ('**', '**')) for m in metric_names]
            for i, r in enumerate(rows)
        ]
        tex_rows = [
            [r['method']] + [_fmt(r[m], i, m, ('\\textbf{', '}')) for m in metric_names]
            for i, r in enumerate(rows)
        ]

        md_path = self.results_dir / 'evaluate.md'
        md_path.write_text(
            f'# Evaluation Results\n\n'
            + tabulate(md_rows, headers=headers, tablefmt='pipe', stralign='right')
            + '\n'
        )
        logger.info('Saved markdown evaluation table to %s', md_path)

        tex_path = self.results_dir / 'evaluate.tex'
        tex_path.write_text(
            tabulate(tex_rows, headers=headers, tablefmt='latex_booktabs', stralign='right') + '\n'
        )
        logger.info('Saved LaTeX evaluation table to %s', tex_path)

    # -------------------------------------------------------------------------
    # Partition-level evaluation
    # -------------------------------------------------------------------------

    def _resolvePartitions(self) -> list[str]:
        p = getattr(self.cfg, 'partition', 'test')
        return ['valid', 'test'] if p == 'all' else [p]

    def _resolveModels(self) -> list[str]:
        m = getattr(self.cfg, 'models', 'all')
        if m == 'all':
            return list(_ALL_MODELS)
        models = [x.strip().upper() for x in m.split(',')]
        unknown = [x for x in models if x not in _KNOWN_MODELS]
        if unknown:
            raise ValueError(f'unknown model(s): {unknown}; valid: {_KNOWN_MODELS}')
        return models

    def _hasFits(self, partition: str) -> bool:
        path = self._partitionDataPath(partition)
        return path.name.endswith('.fit.npz')

    def _getPartitionData(
        self, partition: str, need_fits: bool = True
    ) -> tuple[Dataloader, dict, Path]:
        if need_fits or hasattr(self.cfg, 'data_path_test') or hasattr(self.cfg, 'data_path_valid'):
            self._ensureFitDataloader(partition)
            dl = self.dl_test if partition == 'test' else self.dl_valid
            path = self._partitionDataPath(partition)
        else:
            dl, path = self._baseDataLoader(partition)
        if partition == 'test':
            self._logMemory('Materializing full batch for partition=%s', partition)
            return dl, dl.fullBatch(), path
        self._logMemory('Materializing full batch for partition=%s', partition)
        return dl, dl.fullBatch(), path

    def _getProposalAndMask(
        self, model: str, partition: str, full_batch: dict, dl: Dataloader
    ) -> tuple[Proposal, np.ndarray | None]:
        """Return (proposal, full-batch mask) — mask is None when model covers all datasets."""
        if model == 'MB':
            return self._loadOrSampleMb(partition, dl), None
        elif model == _MB_IMH:
            base = self._loadOrSampleMb(partition, dl)
            return self._refineMb(partition, base, full_batch), None
        elif model == 'NUTS':
            return self._fit2proposal(full_batch, prefix='nuts'), None
        elif model == 'ADVI':
            mask = fitBatchMask(full_batch, prefix='advi')
            return self._fit2proposal(subsetBatch(full_batch, mask), prefix='advi'), mask
        elif model == 'LAPLACE':
            mask = fitBatchMask(full_batch, prefix='laplace')
            return self._fit2proposal(subsetBatch(full_batch, mask), prefix='laplace'), mask
        raise ValueError(f'unknown model: {model}')

    @staticmethod
    def _commonMask(masks: list[np.ndarray | None], n: int) -> np.ndarray | None:
        """Intersection of non-None boolean masks; returns None if all are None."""
        result = np.ones(n, dtype=bool)
        any_mask = False
        for m in masks:
            if m is not None:
                result &= m
                any_mask = True
        return result if any_mask and not result.all() else None

    @staticmethod
    def _alignToCommon(
        proposal: Proposal,
        src_mask: np.ndarray | None,
        common_mask: np.ndarray | None,
    ) -> Proposal:
        """Subset a proposal (indexed by src_mask in full batch) down to common_mask."""
        if common_mask is None:
            return proposal
        if src_mask is None:
            return subsetProposal(proposal, common_mask)
        # common_mask[src_mask]: which of the src positions survive in the common set
        return subsetProposal(proposal, common_mask[src_mask])

    @staticmethod
    def _maskTensor(
        tensor: torch.Tensor,
        mask: np.ndarray | None,
    ) -> torch.Tensor:
        return tensor if mask is None else tensor[torch.from_numpy(mask)]

    def _evalPartitionLight(
        self, partition: str, active: list[str], fit_label: str, multi: bool
    ) -> list[dict]:
        """Evaluate fit comparisons without loading all arrays in ``*.fit.npz`` via Collection."""
        fit_path = self._fitDataPath(partition)
        dl, full_batch, _ = self._getPartitionData(partition, need_fits=False)
        n = full_batch['X'].shape[0]

        masks = {
            model: self._fitMaskFromPath(fit_path, model)
            for model in active
            if model in _FIT_MODELS
        }
        common_mask = self._commonMask(list(masks.values()), n)
        common_batch = (
            subsetBatch(full_batch, common_mask) if common_mask is not None else full_batch
        )

        raw: dict[str, tuple[Proposal, np.ndarray | None]] = {}
        aligned: dict[str, Proposal] = {}
        summaries: dict[str, EvaluationSummary] = {}
        rows: list[dict] = []

        for model in active:
            if model not in _FIT_MODELS:
                continue
            src_mask = masks[model]
            scale = self._maskTensor(full_batch['sd_y'], src_mask) if self.cfg.rescale else None
            proposal = self._fitProposalFromNpz(fit_path, model, mask=src_mask, scale=scale)
            raw[model] = (proposal, src_mask)
            aligned[model] = self._alignToCommon(proposal, src_mask, common_mask)

        for model in active:
            if model in _FIT_MODELS:
                s = self._loadOrComputeSummary(
                    aligned[model],
                    common_batch,
                    partition,
                    model.lower(),
                    mask=self._fitSummaryMask(raw[model][1], common_mask, n),
                )
            elif model in _MB_MODELS:
                method = self._methodName(model)
                s = self._loadCachedSummary(partition, method, mask=common_mask)
                raw[model] = self._getProposalAndMask(model, partition, full_batch, dl)
                aligned[model] = self._alignToCommon(raw[model][0], raw[model][1], common_mask)
                if s is None:
                    s = self._loadOrComputeSummary(
                        aligned[model], common_batch, partition, method, mask=common_mask
                    )
            else:
                raise ValueError(f'unknown model: {model}')
            summaries[model] = s
            label = self._displayLabel(model, partition, multi)
            rows.append(self._makeRow(label, s, fit_label))

        if self.cfg.plot:
            plot_batch = self._plotBatch(common_batch)
            del common_batch, full_batch
            gc.collect()
            self._logMemory('Released full base batch before plotting partition=%s', partition)

            plot_dir = self.plot_dir if partition == 'test' else self.plot_dir / partition
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_models = [model for model in active if model in aligned and model in summaries]
            self.plot(
                [aligned[model] for model in plot_models],
                [summaries[model] for model in plot_models],
                [self._displayModel(model) for model in plot_models],
                plot_batch,
                plot_dir=plot_dir,
            )
        return rows

    def _evalPartition(
        self, partition: str, models: list[str], fit_label: str, multi: bool
    ) -> list[dict]:
        active = self._activeModels(partition, models)

        if not active:
            return []

        need_fits = any(model in _FIT_MODELS for model in active)
        if (
            (self.cfg.plot or self._tableOnlyMode())
            and need_fits
            and not self.cfg.converged_subset
            and not self._directDataMode()
        ):
            logger.info(
                'Using light fit evaluation path for partition=%s; fit arrays are loaded '
                'directly from npz by requested method.',
                partition,
            )
            return self._evalPartitionLight(partition, active, fit_label, multi)

        dl, full_batch, _ = self._getPartitionData(partition, need_fits=need_fits)

        # Collect fit-model proposals first. MB can be skipped entirely when its
        # summary cache is available and no plot/subset proposal is needed.
        raw: dict[str, tuple[Proposal, np.ndarray | None]] = {
            model: self._getProposalAndMask(model, partition, full_batch, dl)
            for model in active
            if model not in _MB_MODELS
        }

        # Align all proposals to their common batch (intersection of all native masks)
        n = full_batch['X'].shape[0]
        common_mask = self._commonMask([mask for _, mask in raw.values()], n)
        common_batch = (
            subsetBatch(full_batch, common_mask) if common_mask is not None else full_batch
        )
        aligned: dict[str, Proposal] = {
            model: self._alignToCommon(proposal, mask, common_mask)
            for model, (proposal, mask) in raw.items()
        }
        for model in active:
            if model in _FIT_MODELS:
                aligned[model] = self._fit2proposal(common_batch, prefix=model.lower())

        # Compute summaries; cache data-derived methods when they are on their native batch
        summaries: dict[str, EvaluationSummary] = {}
        rows: list[dict] = []
        for model in active:
            if model in _FIT_MODELS:
                s = self._loadOrComputeSummary(
                    aligned[model],
                    common_batch,
                    partition,
                    model.lower(),
                    mask=self._fitSummaryMask(raw[model][1], common_mask, n),
                )
            elif model in _MB_MODELS:
                method = self._methodName(model)
                s = self._loadCachedSummary(partition, method, mask=common_mask)
                if s is None:
                    raw[model] = self._getProposalAndMask(model, partition, full_batch, dl)
                    aligned[model] = self._alignToCommon(raw[model][0], raw[model][1], common_mask)
                    s = self._loadOrComputeSummary(
                        aligned[model], common_batch, partition, method, mask=common_mask
                    )
            else:
                s = self.summary(aligned[model], common_batch)
            summaries[model] = s
            label = self._displayLabel(model, partition, multi)
            rows.append(self._makeRow(label, s, fit_label))

        # Comparison plot
        plot_dir = self.plot_dir if partition == 'test' else self.plot_dir / partition
        if self.cfg.plot:
            for model in active:
                if model in _MB_MODELS and model not in aligned:
                    raw[model] = self._getProposalAndMask(model, partition, full_batch, dl)
                    aligned[model] = self._alignToCommon(raw[model][0], raw[model][1], common_mask)
            plot_batch = self._plotBatch(common_batch)
            if not self.cfg.converged_subset:
                del common_batch, full_batch
                gc.collect()
                self._logMemory('Released full fit batch before plotting partition=%s', partition)
            plot_models = [model for model in active if model in aligned and model in summaries]
            plot_dir.mkdir(parents=True, exist_ok=True)
            self.plot(
                [aligned[model] for model in plot_models],
                [summaries[model] for model in plot_models],
                [self._displayModel(model) for model in plot_models],
                plot_batch,
                plot_dir=plot_dir,
            )

        # NUTS convergence diagnostics and sub-population rows
        if self.cfg.converged_subset and 'NUTS' in active:
            for model in active:
                if model in _MB_MODELS and model not in raw:
                    raw[model] = self._getProposalAndMask(model, partition, full_batch, dl)
            rows += self._convergedRows(partition, active, raw, full_batch, fit_label, plot_dir)

        return rows

    def _convergedRows(
        self,
        partition: str,
        active: list[str],
        raw: dict[str, tuple[Proposal, np.ndarray | None]],
        full_batch: dict,
        fit_label: str,
        base_plot_dir: Path,
    ) -> list[dict]:
        """Evaluate NUTS-converged and LOO-reliable subsets; return additional table rows."""
        nuts_proposal, _ = raw['NUTS']
        # Full-batch NUTS summary for diagnostics (always cached)
        summary_nuts_full = self._loadOrComputeSummary(nuts_proposal, full_batch, partition, 'nuts')
        self._nutsFailureAnalysis(summary_nuts_full, full_batch)

        n = full_batch['X'].shape[0]
        conv_mask = nutsConvergeMask(full_batch, mode=self.cfg.convergence_mode)
        if conv_mask is None or not (0 < int(conv_mask.sum()) < n):
            return []

        n_conv = int(conv_mask.sum())
        logger.info('\nConverged subset (%s): %d / %d', self.cfg.convergence_mode, n_conv, n)

        rows = self._subsetEval(
            'conv',
            active,
            raw,
            full_batch,
            conv_mask,
            fit_label,
            do_plot=True,
            base_plot_dir=base_plot_dir,
        )

        # LOO-reliable subset: NUTS Pareto-k filter applied on top of convergence
        nuts_k = summary_nuts_full.per_dataset.loo_pareto_k
        if nuts_k is not None:
            k_thr = self.cfg.pareto_k_thr
            k_mask = (nuts_k < k_thr).numpy() & conv_mask
            n_k = int(k_mask.sum())
            logger.info('Reliable LOO subset (k<%.1f): %d / %d', k_thr, n_k, n_conv)
            if 0 < n_k < n_conv:
                rows += self._subsetEval(
                    'loo',
                    active,
                    raw,
                    full_batch,
                    k_mask,
                    fit_label,
                    do_plot=False,
                    base_plot_dir=base_plot_dir,
                )

        return rows

    def _subsetEval(
        self,
        tag: str,
        active: list[str],
        raw: dict[str, tuple[Proposal, np.ndarray | None]],
        full_batch: dict,
        subset_mask: np.ndarray,
        fit_label: str,
        do_plot: bool = False,
        base_plot_dir: Path | None = None,
    ) -> list[dict]:
        """Evaluate active models on the subset of full_batch selected by subset_mask."""
        n = full_batch['X'].shape[0]

        # Re-index each model's proposal into the subset_mask context
        sub_raw: dict[str, tuple[Proposal, np.ndarray | None]] = {}
        for model, (proposal, src_mask) in raw.items():
            if src_mask is None:
                sub_raw[model] = (subsetProposal(proposal, subset_mask), subset_mask)
            else:
                new_mask = src_mask & subset_mask
                sub_raw[model] = (subsetProposal(proposal, subset_mask[src_mask]), new_mask)

        # Align within the subset context (handles any model with a narrower native mask)
        sub_common_mask = self._commonMask([mask for _, mask in sub_raw.values()], n)
        sub_common_batch = subsetBatch(full_batch, sub_common_mask)
        sub_aligned: dict[str, Proposal] = {
            model: self._alignToCommon(proposal, mask, sub_common_mask)
            for model, (proposal, mask) in sub_raw.items()
        }

        summaries: dict[str, EvaluationSummary] = {}
        rows: list[dict] = []
        for model in active:
            s = self.summary(sub_aligned[model], sub_common_batch)
            summaries[model] = s
            rows.append(self._makeRow(f'{self._displayModel(model)}_{tag}', s, fit_label))

        if do_plot and len(active) > 1:
            plot_dir = (base_plot_dir or self.plot_dir) / tag
            plot_dir.mkdir(parents=True, exist_ok=True)
            self.plot(
                list(sub_aligned.values()),
                list(summaries.values()),
                [self._displayModel(model) for model in active],
                sub_common_batch,
                plot_dir=plot_dir,
            )

        return rows

    def _nutsFailureAnalysis(
        self,
        summary: EvaluationSummary,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Report NUTS convergence diagnostics and their Spearman correlation with LOO-NLL."""
        if 'nuts_divergences' not in batch:
            return

        from scipy.stats import spearmanr

        def _nanfield(key: str) -> np.ndarray | None:
            return batch[key].numpy().astype(np.float64) if key in batch else None

        def _param_stat(arr, fn):
            if arr is None:
                return None
            a = arr.copy()
            a[a <= 0] = np.nan
            return fn(a, axis=-1)

        conv = nutsConvergeMask(batch, mode=self.cfg.convergence_mode)
        fail = ~conv
        total_div = batch['nuts_divergences'].numpy().sum(-1)
        duration = batch['nuts_duration'].numpy().ravel()

        ess_tail = _nanfield('nuts_ess_tail')
        rhat = _nanfield('nuts_rhat')
        treedepth = _nanfield('nuts_max_treedepth')

        min_ess = _param_stat(_nanfield('nuts_ess'), np.nanmin)
        min_ess_tail = _param_stat(ess_tail, np.nanmin)
        max_rhat = _param_stat(rhat, np.nanmax)
        mean_treedepth_sat = treedepth.mean(-1) if treedepth is not None else None

        loo = (
            summary.per_dataset.loo_nll.numpy() if summary.per_dataset.loo_nll is not None else None
        )
        b = len(total_div)

        f_rhat = (max_rhat > 1.01) if max_rhat is not None else np.zeros(b, bool)
        f_div = total_div > 0
        f_tree = (
            (mean_treedepth_sat > 0.05) if mean_treedepth_sat is not None else np.zeros(b, bool)
        )
        f_ess = (min_ess < 400) if min_ess is not None else np.zeros(b, bool)
        f_ess_tail = (min_ess_tail < 400) if min_ess_tail is not None else np.zeros(b, bool)

        counts = {
            'R-hat > 1.01': int(f_rhat.sum()),
            'divergences > 0': int(f_div.sum()),
            'tree-depth sat > 5%': int(f_tree.sum()),
            'ESS < 400': int(f_ess.sum()),
            'tail ESS < 400': int(f_ess_tail.sum()),
            'any failure': int(fail.sum()),
        }

        diag_pairs = [
            ('Max R-hat', max_rhat),
            ('Total divergences', total_div),
            ('Mean tree-depth sat', mean_treedepth_sat),
            ('Min ESS (bulk)', min_ess),
            ('Min ESS (tail)', min_ess_tail),
            ('Duration [s]', duration),
        ]
        corr_rows = []
        if loo is not None:
            for name, diag in diag_pairs:
                if diag is None:
                    continue
                ok = np.isfinite(diag) & np.isfinite(loo)
                r_s = (
                    float(spearmanr(diag[ok], loo[ok]).statistic) if ok.sum() > 2 else float('nan')
                )
                corr_rows.append([name, r_s])

        lines = ['  ' + '  |  '.join(f'{k}: {v}/{b}' for k, v in counts.items())]
        if loo is not None and fail.any() and (~fail).any():
            fail_med = float(np.median(loo[fail]))
            clean_med = float(np.median(loo[~fail]))
            lines.append(
                f'  Median LOO-NLL:  {fail_med:.3f} (fail) vs {clean_med:.3f} (clean)'
                f'   Δ = {fail_med - clean_med:+.3f}'
            )

        corr_table = tabulate(
            corr_rows, headers=['Diagnostic', 'ρ(LOO-NLL)'], floatfmt='.3f', tablefmt='simple'
        )
        logger.info('\nNUTS diagnostics (%d datasets)\n%s\n%s\n', b, corr_table, '\n'.join(lines))

    # -------------------------------------------------------------------------
    # Entry points
    # -------------------------------------------------------------------------

    def go(self) -> None:
        partitions = self._resolvePartitions()
        models = self._resolveModels()
        fit_label = self._fitLabel()
        multi = len(partitions) > 1
        rows: list[dict] = []
        for partition in partitions:
            cached_rows = None
            if self._tableOnlyMode():
                cached_rows = self._cachedRowsForPartition(partition, models, fit_label, multi)
                if (
                    cached_rows is None
                    and not self._canFallbackFromTableOnly(partition, models)
                    and not self._canLightEvaluateFromTableOnly(partition, models)
                ):
                    raise self._tableOnlyCacheError(partition)
            if cached_rows is not None:
                rows.extend(cached_rows)
                continue
            rows.extend(self._evalPartition(partition, models, fit_label, multi))
        if self.cfg.save_tables and rows:
            self.saveTables(rows)


# =============================================================================
def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    if not getattr(cfg, 'show', False):
        # batch runs must never block on a GUI window waiting to be closed
        matplotlib.use('Agg', force=True)
    try:
        evaluator = Evaluator(cfg)
        evaluator.go()
        logger.info('Evaluation finished successfully.')
    except MemoryError:
        logger.exception(
            'Evaluation failed with Python MemoryError [max RSS %.1f MiB]. '
            'For cached table generation, use --no-plot --save_tables so evaluate.py can avoid '
            'loading full fit files. For uncached summaries, lower --batch_size, --n_samples, or '
            '--summary_chunk_size.',
            Evaluator._maxRssMb(),
        )
        raise
    except RuntimeError as exc:
        if 'out of memory' in str(exc).lower():
            logger.exception(
                'Evaluation failed with RuntimeError out-of-memory [max RSS %.1f MiB]. '
                'Try --summary_chunk_size 1 for predictive coverage, reduce --batch_size or '
                '--n_samples, or generate tables from cached summaries with --no-plot.',
                Evaluator._maxRssMb(),
            )
        raise


if __name__ == '__main__':
    main()
