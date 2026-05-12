import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.utils.logger import setupLogging
from metabeta.utils.io import setDevice, datasetFilename, runName
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
    Proposal,
    concatProposalsBatch,
    dictMean,
    nutsConvergeMask,
    subsetProposal,
)
from metabeta.models.approximator import Approximator
from metabeta.utils.moe import moeEstimate
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.plotting import plotComparison

logger = logging.getLogger('evaluate.py')

_ALL_MODELS = ('MB', 'NUTS', 'ADVI')
_FIT_MODELS = frozenset(('NUTS', 'ADVI'))


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # Primary: load from checkpoint (required when MB is evaluated)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--prefix', type=str, default='latest', help='Checkpoint prefix: best or latest')
    # Data-direct: evaluate fit-only models without a checkpoint
    parser.add_argument('--data_path_test', type=str, help='Direct path to test.fit.npz (no checkpoint needed for NUTS/ADVI)')
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
    parser.add_argument('--conformal', action=argparse.BooleanOptionalAction)
    parser.add_argument('--k', type=int, default=0, help='pseudo-MoE permuted views (0=off)')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_tables', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--outdir', type=str)
    parser.add_argument(
        '--partition', type=str, default='test', choices=['valid', 'test', 'all'],
        help='Data partition(s) to evaluate: valid, test, or all (default: test)',
    )
    parser.add_argument(
        '--models', type=str, default='all',
        help='Models to evaluate: comma-separated MB/NUTS/ADVI or "all" (default: all)',
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
        '--comparison_legend', type=str, choices=['panel', 'right'], default='panel',
        help='Comparison plot legend placement (default: panel)',
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
        if 'MB' in active:
            raise ValueError(
                '--data_path_test/--data_path_valid mode: use --models NUTS,ADVI (no MB without --checkpoint)'
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
        self.cfg.k = getattr(cfg, 'k', 0)
        self.cfg.save_tables = getattr(cfg, 'save_tables', False)
        self.cfg.converged_subset = getattr(cfg, 'converged_subset', False)
        self.cfg.convergence_mode = getattr(cfg, 'convergence_mode', 'liberal')
        self.cfg.pareto_k_thr = getattr(cfg, 'pareto_k_thr', 0.7)
        self.cfg.outdir = getattr(cfg, 'outdir', str(Path(self.dir, '..', 'outputs', 'results')))

        if hasattr(cfg, 'data_path_test') or hasattr(cfg, 'data_path_valid'):
            # data-direct mode: run name derived from the data directory
            data_p = Path(getattr(cfg, 'data_path_test', None) or cfg.data_path_valid)
            self.run_name = data_p.parent.name
            self.ckpt_dir = None
        else:
            self.run_name = runName(vars(cfg))
            if hasattr(cfg, '_checkpoint_dir'):
                self.ckpt_dir = Path(cfg._checkpoint_dir)
            else:
                self.ckpt_dir = Path(self.dir, '..', 'outputs', 'checkpoints', self.run_name)
        self.checkpoint_prefix = getattr(cfg, 'prefix', 'latest')

        self._initData()
        if 'MB' in self._resolveModels():
            self._initModel()
            self._load()

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

    def _initDataFromConfig(self) -> None:
        self.data_cfg_train = loadDataConfig(self.cfg.data_id)
        assimilateConfig(self.cfg, self.data_cfg_train)
        self.cfg.data_id_valid = getattr(self.cfg, 'data_id_valid', self.cfg.data_id)
        self.cfg.data_id_test = getattr(self.cfg, 'data_id_test', self.cfg.data_id_valid)
        self.data_cfg_valid = loadDataConfig(self.cfg.data_id_valid)
        self.data_cfg_test = loadDataConfig(self.cfg.data_id_test)
        self.data_cfg = self.data_cfg_train
        self.dl_valid, self.data_path_valid = self._getDataLoader(
            'valid', batch_size=self.cfg.batch_size
        )
        self.dl_test, self.data_path_test = self._getDataLoader(
            'test', batch_size=self.cfg.batch_size
        )

    def _initDataDirect(self) -> None:
        """Initialise data from explicit file paths; infers config fields from the npz."""
        test_p = Path(self.cfg.data_path_test) if hasattr(self.cfg, 'data_path_test') else None
        valid_p = Path(self.cfg.data_path_valid) if hasattr(self.cfg, 'data_path_valid') else None
        self.data_path_test = test_p or valid_p
        self.data_path_valid = valid_p or test_p

        bs = self.cfg.batch_size
        self.dl_test = Dataloader(self.data_path_test, batch_size=bs, sortish=True)
        self.dl_valid = Dataloader(self.data_path_valid, batch_size=bs, sortish=True)

        # Infer missing config fields from the collection
        col = self.dl_test.dataset
        if not hasattr(self.cfg, 'max_d'):
            self.cfg.max_d = col.d
        if not hasattr(self.cfg, 'max_q'):
            self.cfg.max_q = col.q
        if not hasattr(self.cfg, 'likelihood_family'):
            raw_lf = col.raw.get('likelihood_family')
            self.cfg.likelihood_family = int(raw_lf[0]) if raw_lf is not None else 0
        if not hasattr(self.cfg, 'rescale'):
            self.cfg.rescale = False

        self.data_cfg = {}

    def _getDataLoader(
        self, partition: str, batch_size: int | None = None
    ) -> tuple[Dataloader, Path]:
        data_cfg = self.data_cfg_test if partition == 'test' else self.data_cfg_valid
        data_fname = datasetFilename(partition)
        data_subdir = data_cfg['data_id']
        data_path = Path(self.dir, '..', 'outputs', 'data', data_subdir, data_fname)
        fit_path = data_path.with_suffix('.fit.npz')
        if fit_path.exists():
            data_path = fit_path
        assert data_path.exists(), f'data file not found: {data_path}'
        sortish = batch_size is not None
        dl = Dataloader(
            data_path,
            batch_size=batch_size,
            sortish=sortish,
            max_d=self.cfg.max_d,
            max_q=self.cfg.max_q,
        )
        return dl, data_path

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

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def _fit2proposal(self, batch: dict[str, torch.Tensor], prefix: str) -> Proposal:
        proposed = {}
        ffx = batch[f'{prefix}_ffx']
        sigma_rfx = batch[f'{prefix}_sigma_rfx']
        samples_g = [ffx, sigma_rfx]
        if f'{prefix}_sigma_eps' in batch:
            sigma_eps = batch[f'{prefix}_sigma_eps'].unsqueeze(-1)
            samples_g.append(sigma_eps)
            has_sigma_eps = True
        else:
            has_sigma_eps = False
        proposed['global'] = {'samples': torch.cat(samples_g, dim=-1)}
        proposed['local'] = {'samples': batch[f'{prefix}_rfx']}
        corr_rfx = batch.get(f'{prefix}_corr_rfx', None)
        proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps, corr_rfx=corr_rfx)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        proposal.tpd = batch[f'{prefix}_duration'].mean().item()
        return proposal

    def _fitBatchMask(self, batch: dict[str, torch.Tensor], prefix: str) -> np.ndarray:
        failed_key = f'{prefix}_failed'
        if failed_key not in batch:
            return np.ones(batch['X'].shape[0], dtype=bool)
        return ~batch[failed_key].cpu().numpy().astype(bool)

    def _sampleBatch(self, batch: dict[str, torch.Tensor]) -> Proposal:
        proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        return proposal

    def _sampleMoe(self, batch: dict[str, torch.Tensor], n_datasets_seen: int) -> list[Proposal]:
        B = batch['X'].shape[0]
        proposals = []
        for i in range(B):
            single = {k: v[i : i + 1] if torch.is_tensor(v) else v for k, v in batch.items()}
            rng = np.random.default_rng(self.cfg.seed + n_datasets_seen + i)
            proposal = moeEstimate(self.model, single, self.cfg.n_samples, self.cfg.k, rng=rng)
            if self.cfg.rescale:
                proposal.rescale(single['sd_y'])
            proposals.append(proposal)
        return proposals

    @torch.no_grad()
    def sample(self, batch: dict[str, torch.Tensor]) -> Proposal:
        batch = toDevice(batch, self.device)
        t0 = time.perf_counter()
        if self.cfg.k > 0:
            proposals = self._sampleMoe(batch, 0)
            proposal = concatProposalsBatch(proposals)
        else:
            proposal = self._sampleBatch(batch)
        t1 = time.perf_counter()
        proposal.tpd = (t1 - t0) / batch['X'].shape[0]
        return proposal

    @torch.no_grad()
    def sampleMinibatched(self, dl: Dataloader, label: str) -> Proposal:
        proposals = []
        n_datasets = 0
        t0 = time.perf_counter()
        for batch in tqdm(dl, desc=f'  {label}'):
            batch = toDevice(batch, self.device)
            if self.cfg.k > 0:
                batch_proposals = self._sampleMoe(batch, n_datasets)
                for p in batch_proposals:
                    p.to('cpu')
                    proposals.append(p)
            else:
                proposal = self._sampleBatch(batch)
                proposal.to('cpu')
                proposals.append(proposal)
            n_datasets += batch['X'].shape[0]
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
            proposal, batch, likelihood_family=lf, compute_pred_coverage=pred_cov
        )
        logger.info(summaryTable(eval_summary, lf))
        return eval_summary

    def _summaryCachePath(self, partition: str, method: str) -> Path:
        data_path = self.data_path_test if partition == 'test' else self.data_path_valid
        return data_path.parent / f'summary_{partition}_{method}.pt'

    def _loadOrComputeSummary(
        self,
        proposal: Proposal,
        batch: dict[str, torch.Tensor],
        partition: str,
        method: str,
    ) -> EvaluationSummary:
        cache_path = self._summaryCachePath(partition, method)
        data_path = self.data_path_test if partition == 'test' else self.data_path_valid
        ref_mtime = data_path.stat().st_mtime if data_path.exists() else 0.0
        if cache_path.exists() and cache_path.stat().st_mtime >= ref_mtime:
            logger.info('Loading cached %s/%s summary from %s', partition, method, cache_path)
            return EvaluationSummary.load(cache_path)
        result = self.summary(proposal, batch)
        result.save(cache_path)
        logger.info('Saved %s/%s summary to %s', partition, method, cache_path)
        return result

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
        target_dir = plot_dir if plot_dir is not None else self.plot_dir
        plotComparison(
            summaries,
            proposals,
            labels,
            batch,
            plot_dir=target_dir,
            show=True,
            legend_right=getattr(self.cfg, 'comparison_legend', 'panel') == 'right',
        )

    def _fitLabel(self) -> str:
        return {0: 'ppR2', 1: 'ppAUC', 2: 'ppDev'}.get(self.cfg.likelihood_family, 'ppR2')

    def _makeRow(self, label: str, summary: EvaluationSummary, fit_label: str) -> dict:
        ag, pd = summary.aggregated, summary.per_dataset
        return {
            'method': label,
            'R': dictMean(ag.corr),
            'NRMSE': dictMean(ag.nrmse),
            'ECE': dictMean(ag.ece),
            'RFX_joint_ECE': ag.rfx_joint_ece,
            'RFX_joint_EACE': ag.rfx_joint_eace,
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
            'RFX_joint_ECE': 'abs',
            'RFX_joint_EACE': False,
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

        def _fmt(val: float | None, i: int, metric: str) -> str:
            if val is None:
                return 'NA'
            cell = f'{val:.4f}'
            return f'**{cell}**' if i in best.get(metric, set()) else cell

        def _fmt_tex(val: float | None, i: int, metric: str) -> str:
            if val is None:
                return 'NA'
            cell = f'{val:.4f}'
            return f'\\textbf{{{cell}}}' if i in best.get(metric, set()) else cell

        headers = ['Method'] + metric_names
        md_rows = [
            [r['method']] + [_fmt(r[m], i, m) for m in metric_names] for i, r in enumerate(rows)
        ]
        tex_rows = [
            [r['method']] + [_fmt_tex(r[m], i, m) for m in metric_names] for i, r in enumerate(rows)
        ]

        md_path = self.results_dir / 'evaluate.md'
        md_path.write_text(
            f'# Evaluation Results\n\n'
            + tabulate(md_rows, headers=headers, tablefmt='pipe', stralign='right')
            + '\n'
        )

        tex_path = self.results_dir / 'evaluate.tex'
        tex_path.write_text(
            tabulate(tex_rows, headers=headers, tablefmt='latex_booktabs', stralign='right') + '\n'
        )

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
        unknown = [x for x in models if x not in _ALL_MODELS]
        if unknown:
            raise ValueError(f'unknown model(s): {unknown}; valid: {_ALL_MODELS}')
        return models

    def _hasFits(self, partition: str) -> bool:
        path = self.data_path_test if partition == 'test' else self.data_path_valid
        return path.name.endswith('.fit.npz')

    def _getPartitionData(self, partition: str) -> tuple[Dataloader, dict, Path]:
        if partition == 'test':
            return self.dl_test, self.dl_test.fullBatch(), self.data_path_test
        return self.dl_valid, self.dl_valid.fullBatch(), self.data_path_valid

    def _getProposalAndMask(
        self, model: str, partition: str, full_batch: dict, dl: Dataloader
    ) -> tuple[Proposal, np.ndarray | None]:
        """Return (proposal, full-batch mask) — mask is None when model covers all datasets."""
        if model == 'MB':
            return self.sampleMinibatched(dl, f'MB ({partition})'), None
        elif model == 'NUTS':
            return self._fit2proposal(full_batch, prefix='nuts'), None
        elif model == 'ADVI':
            mask = self._fitBatchMask(full_batch, prefix='advi')
            return self._fit2proposal(subsetBatch(full_batch, mask), prefix='advi'), mask
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
        return result if any_mask else None

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

    def _evalPartition(
        self, partition: str, models: list[str], fit_label: str, multi: bool
    ) -> list[dict]:
        dl, full_batch, _ = self._getPartitionData(partition)
        has_fits = self._hasFits(partition)
        active = [m for m in models if m == 'MB' or (has_fits and m in _FIT_MODELS)]

        if not active:
            logger.warning('No active models for partition=%s (no fit file found)', partition)
            return []

        # Collect raw (native-batch) proposals
        raw: dict[str, tuple[Proposal, np.ndarray | None]] = {
            model: self._getProposalAndMask(model, partition, full_batch, dl) for model in active
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

        # Compute summaries; cache data-derived methods when they are on their native batch
        summaries: dict[str, EvaluationSummary] = {}
        rows: list[dict] = []
        for model in active:
            native_batch = model in _FIT_MODELS and (
                model == 'ADVI' or (model == 'NUTS' and common_mask is None)
            )
            if native_batch:
                s = self._loadOrComputeSummary(
                    aligned[model], common_batch, partition, model.lower()
                )
            else:
                s = self.summary(aligned[model], common_batch)
            summaries[model] = s
            label = f'{model}_{partition}' if multi else model
            rows.append(self._makeRow(label, s, fit_label))

        # Comparison plot
        plot_dir = self.plot_dir if partition == 'test' else self.plot_dir / partition
        plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot(
            list(aligned.values()),
            list(summaries.values()),
            active,
            common_batch,
            plot_dir=plot_dir,
        )

        # NUTS convergence diagnostics and sub-population rows
        if self.cfg.converged_subset and has_fits and 'NUTS' in active:
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
            rows.append(self._makeRow(f'{model}_{tag}', s, fit_label))

        if do_plot and len(active) > 1:
            plot_dir = (base_plot_dir or self.plot_dir) / tag
            plot_dir.mkdir(parents=True, exist_ok=True)
            self.plot(
                list(sub_aligned.values()),
                list(summaries.values()),
                active,
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

    def testrun(self) -> None:
        full_batch = self.dl_valid.fullBatch()
        proposal_mb = self.sampleMinibatched(self.dl_valid, 'MB')
        summary_mb = self.summary(proposal_mb, full_batch)
        self.plot([proposal_mb], [summary_mb], ['MB'], full_batch)

    def go(self) -> None:
        partitions = self._resolvePartitions()
        models = self._resolveModels()
        fit_label = self._fitLabel()
        multi = len(partitions) > 1
        rows: list[dict] = []
        for partition in partitions:
            rows.extend(self._evalPartition(partition, models, fit_label, multi))
        if self.cfg.save_tables:
            self.saveTables(rows)


# =============================================================================
def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    evaluator = Evaluator(cfg)
    evaluator.go()


if __name__ == '__main__':
    main()
