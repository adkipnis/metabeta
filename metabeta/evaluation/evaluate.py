import yaml
import time
import logging
import argparse
from pathlib import Path

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
    dictMeanExcl,
    nutsConvergeMask,
    subsetProposal,
)
import numpy as np

from metabeta.models.approximator import Approximator
from metabeta.utils.moe import moeEstimate
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.plotting import plotComparison

logger = logging.getLogger('evaluate.py')


def setup() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Primary: Load from checkpoint config
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument(
        '--prefix', type=str, default='latest', help='Checkpoint prefix: best or latest'
    )

    # Legacy: Load from config file
    parser.add_argument('--config', type=str, help='Path to custom YAML config file')

    # Config overrides
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--r_tag', type=str)
    parser.add_argument('--data_id', type=str)
    parser.add_argument('--data_id_valid', type=str)

    # CLI-only runtime params
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbosity', type=int, default=1)

    # Evaluation settings (override checkpoint config)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--conformal', action=argparse.BooleanOptionalAction)
    parser.add_argument('--k', type=int, default=0, help='pseudo-MoE permuted views (0=off)')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_tables', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--outdir', type=str)
    parser.add_argument(
        '--converged_subset',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Also evaluate/plot on the NUTS-converged subset of test datasets',
    )
    parser.add_argument(
        '--convergence_mode',
        type=str,
        default='liberal',
        choices=['strict', 'liberal'],
        help='NUTS convergence filter mode for converged_subset (default: liberal)',
    )
    parser.add_argument(
        '--pareto_k_thr',
        type=float,
        default=0.7,
        help='Pareto-k threshold for LOO-NLL subset; filters on NUTS k only (default: 0.7)',
    )
    parser.add_argument(
        '--pred_coverage',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Compute predictive interval coverage/width (adds Pred. EACE and 90%% width to output)',
    )

    args = parser.parse_args()
    args.config = Path(
        '..', 'outputs', 'checkpoints', 'normal_dsmall-n-mixed_mlarge_s0', 'config.yaml'
    )
    # args.config = Path('..', 'outputs', 'checkpoints', 'normal_dmedium-n-mixed_mlarge_s2', 'config.yaml')
    # args.config = Path('..', 'outputs', 'checkpoints', 'normal_dmedium-n-mixed_mhuge-s_s1', 'config.yaml')

    # Load config from checkpoint or file
    if hasattr(args, 'checkpoint') and args.checkpoint:
        # Load from checkpoint directory
        checkpoint_path = Path(args.checkpoint)
        cfg_dict = loadConfigFromCheckpoint(checkpoint_path)
        # Store checkpoint path and prefix for loading model
        cfg_dict['_checkpoint_dir'] = str(checkpoint_path)
        cfg_dict['_checkpoint_prefix'] = args.prefix
        # Merge CLI overrides
        for k, v in vars(args).items():
            if v is not None and k not in ['checkpoint', 'prefix', 'config', 'name']:
                cfg_dict[k] = v
    elif hasattr(args, 'config') and args.config:
        # Custom YAML config
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        # Merge CLI args
        for k, v in vars(args).items():
            if v is not None and k not in ['checkpoint', 'config', 'name']:
                cfg_dict[k] = v
    else:
        raise ValueError(
            'Must specify one of:\n'
            '  1. Checkpoint: --checkpoint <checkpoint_dir> [--prefix best|latest]\n'
            '  2. Custom config: --config <path>\n'
        )

    if cfg_dict.get('save_tables') is None:
        cfg_dict['save_tables'] = True

    return argparse.Namespace(**cfg_dict)


# -----------------------------------------------------------------------------
class Evaluator:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.dir = Path(__file__).resolve().parent
        setSeed(cfg.seed)
        self.device = setDevice(cfg.device)

        if not hasattr(self.cfg, 'batch_size'):
            self.cfg.batch_size = 8
        if not hasattr(self.cfg, 'k'):
            self.cfg.k = 0
        if not hasattr(self.cfg, 'save_tables'):
            self.cfg.save_tables = False
        if not hasattr(self.cfg, 'converged_subset'):
            self.cfg.converged_subset = False
        if not hasattr(self.cfg, 'convergence_mode'):
            self.cfg.convergence_mode = 'liberal'
        if not hasattr(self.cfg, 'pareto_k_thr'):
            self.cfg.pareto_k_thr = 0.7
        if not hasattr(self.cfg, 'outdir'):
            self.cfg.outdir = str(Path(self.dir, '..', 'outputs', 'results'))

        # checkpoint dir
        if hasattr(self.cfg, '_checkpoint_dir'):
            # Use explicit checkpoint directory from --checkpoint arg
            self.ckpt_dir = Path(self.cfg._checkpoint_dir)
        else:
            # Legacy: construct from run name
            self.run_name = runName(vars(self.cfg))
            self.ckpt_dir = Path(self.dir, '..', 'outputs', 'checkpoints', self.run_name)
        self.checkpoint_prefix = getattr(self.cfg, 'prefix', 'latest')

        # load data and model
        self._initData()
        self._initModel()
        self._load()

        # plot dir
        self.plot_dir = None
        self.plot_dir = Path(self.dir, '..', 'outputs', 'plots', self.run_name)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # results dir
        self.results_dir = None
        if self.cfg.save_tables:
            base_dir = Path(self.cfg.outdir)
            self.results_dir = base_dir / self.run_name
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def _initData(self) -> None:
        # assimilate training data config for model/checkpoint consistency
        self.data_cfg_train = loadDataConfig(self.cfg.data_id)
        assimilateConfig(self.cfg, self.data_cfg_train)

        # allow overriding validation/test data id independently from training
        self.data_cfg_valid = loadDataConfig(self.cfg.data_id_valid)

        # keep legacy attr name for checkpoint comparison compatibility
        self.data_cfg = self.data_cfg_train

        # get dataloaders
        self.dl_valid = self._getDataLoader('valid', batch_size=self.cfg.batch_size)
        self.dl_test = self._getDataLoader('test', batch_size=self.cfg.batch_size)

    def _getDataLoader(self, partition: str, batch_size: int | None = None) -> Dataloader:
        data_cfg = self.data_cfg_valid
        data_fname = datasetFilename(partition)
        data_subdir = data_cfg['data_id']
        data_path = Path(self.dir, '..', 'outputs', 'data', data_subdir, data_fname)
        if partition == 'test':
            data_path = data_path.with_suffix('.fit.npz')
        assert data_path.exists(), f'data file not found: {data_path}'
        sortish = batch_size is not None
        return Dataloader(data_path, batch_size=batch_size, sortish=sortish)

    def _initModel(self) -> None:
        """Load model architecture from config and restore checkpoint weights."""
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

        # compare configs
        if self.data_cfg != payload['data_cfg']:
            logger.warning('data config mismatch between current and checkpoint')
        if self.model_cfg.to_dict() != payload['model_cfg']:
            logger.warning('model config mismatch between current and checkpoint')

        # load states
        self.model.load_state_dict(payload['model_state'])

        # optionally compile
        if self.cfg.compile and self.device.type == 'cuda':
            self.model.compile()

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
        """Sample a proposal from a batch (no MoE)."""
        proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        return proposal

    def _sampleMoe(self, batch: dict[str, torch.Tensor], n_datasets_seen: int) -> list[Proposal]:
        """Sample with pseudo-MoE (B=1 per dataset), optionally applying IS."""
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

    @torch.inference_mode()
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

    @torch.inference_mode()
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
        summary_table = summaryTable(eval_summary, lf)
        logger.info(summary_table)
        return eval_summary

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
        plotComparison(summaries, proposals, labels, batch, plot_dir=target_dir, show=True)

    def testrun(self) -> None:
        full_batch = self.dl_valid.fullBatch()
        proposal_mb = self.sampleMinibatched(self.dl_valid, 'MB')
        summary_mb = self.summary(proposal_mb, full_batch)
        self.plot([proposal_mb], [summary_mb], ['MB'], full_batch)

    def _fitLabel(self) -> str:
        labels = {0: 'ppR2', 1: 'ppAUC', 2: 'ppDev'}
        return labels.get(self.cfg.likelihood_family, 'ppR2')

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

        table_rows = []
        for i, r in enumerate(rows):
            row = [r['method']]
            for metric in metric_names:
                val = r[metric]
                if val is None:
                    cell = 'NA'
                else:
                    cell = f'{val:.4f}'
                    if i in best.get(metric, set()):
                        cell = f'**{cell}**'
                row.append(cell)
            table_rows.append(row)

        headers = ['Method'] + metric_names
        md_table = tabulate(table_rows, headers=headers, tablefmt='pipe', stralign='right')
        md_path = self.results_dir / 'evaluate.md'
        md_path.write_text(f'# Evaluation Results\n\n{md_table}\n')

        table_rows_tex = []
        for i, r in enumerate(rows):
            row = [r['method']]
            for metric in metric_names:
                val = r[metric]
                if val is None:
                    cell = 'NA'
                else:
                    cell = f'{val:.4f}'
                    if i in best.get(metric, set()):
                        cell = f'\\textbf{{{cell}}}'
                row.append(cell)
            table_rows_tex.append(row)

        tex_table = tabulate(
            table_rows_tex, headers=headers, tablefmt='latex_booktabs', stralign='right'
        )
        tex_path = self.results_dir / 'evaluate.tex'
        tex_path.write_text(tex_table + '\n')

    def _nutsFailureAnalysis(
        self,
        summary: EvaluationSummary,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Report NUTS convergence diagnostics and their Spearman correlation with LOO-NLL.

        Failure criteria (in descending severity):
          1. R-hat > 1.01 on any parameter  — chains didn't converge to the same distribution
          2. Any divergences                 — sampler hit a region of bad geometry
          3. >5% tree-depth saturation      — trajectories truncated, stationary dist. may be wrong
          4. Min bulk or tail ESS < 400     — too few effective samples for reliable inference
        """
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

        loo = summary.loo_nll.numpy() if summary.loo_nll is not None else None
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

        # --- Spearman correlations with LOO-NLL ---
        diag_pairs = [
            ('Max R-hat', max_rhat),
            ('Total divergences', total_div),
            ('Mean tree-depth sat', mean_treedepth_sat),
            ('Min ESS (bulk)', min_ess),
            ('Min ESS (tail)', min_ess_tail),
            ('Duration [s]', duration),
        ]
        rows = []
        if loo is not None:
            for name, diag in diag_pairs:
                if diag is None:
                    continue
                ok = np.isfinite(diag) & np.isfinite(loo)
                r_s = (
                    float(spearmanr(diag[ok], loo[ok]).statistic) if ok.sum() > 2 else float('nan')
                )
                rows.append([name, r_s])

        # --- failure vs clean LOO-NLL ---
        lines = ['  ' + '  |  '.join(f'{k}: {v}/{b}' for k, v in counts.items())]
        if loo is not None and fail.any() and (~fail).any():
            fail_med = float(np.median(loo[fail]))
            clean_med = float(np.median(loo[~fail]))
            lines.append(
                f'  Median LOO-NLL:  {fail_med:.3f} (fail) vs {clean_med:.3f} (clean)'
                f'   Δ = {fail_med - clean_med:+.3f}'
            )

        corr_table = tabulate(
            rows, headers=['Diagnostic', 'ρ(LOO-NLL)'], floatfmt='.3f', tablefmt='simple'
        )
        logger.info('\nNUTS diagnostics (%d datasets)\n%s\n%s\n', b, corr_table, '\n'.join(lines))

    def _makeRow(self, label: str, summary: EvaluationSummary, fit_label: str) -> dict:
        return {
            'method': label,
            'R': dictMeanExcl(summary.corr),
            'NRMSE': dictMeanExcl(summary.nrmse),
            'ECE': dictMeanExcl(summary.ece),
            'RFX_joint_ECE': summary.rfx_joint_ece,
            'RFX_joint_EACE': summary.rfx_joint_eace,
            'ppNLL': summary.mnll,
            fit_label: summary.mfit,
            'tpd': summary.tpd,
            'IS_eff': summary.meff,
            'Pareto_k': summary.mk,
            'ppEACE': summary.pp_eace,
            'ppWidth90': summary.pp_width_90,
        }

    def go(self) -> None:
        full_batch = self.dl_test.fullBatch()
        advi_mask = self._fitBatchMask(full_batch, prefix='advi')
        advi_batch = subsetBatch(full_batch, advi_mask)

        # MB proposal
        proposal_mb = self.sampleMinibatched(self.dl_test, 'MB')
        summary_mb = self.summary(proposal_mb, full_batch)

        # NUTS proposal
        proposal_nuts = self._fit2proposal(full_batch, prefix='nuts')
        summary_nuts = self.summary(proposal_nuts, full_batch)
        self._nutsFailureAnalysis(summary_nuts, full_batch)

        # ADVI proposal
        proposal_advi = self._fit2proposal(advi_batch, prefix='advi')
        summary_advi = self.summary(proposal_advi, advi_batch)

        self.plot([proposal_mb, proposal_nuts], [summary_mb, summary_nuts], ['MB', 'NUTS'], full_batch)
        self.plot([proposal_advi], [summary_advi], ['ADVI'], advi_batch, plot_dir=self.plot_dir / 'advi')

        fit_label = self._fitLabel()
        rows = []
        for label, summary in [
            ('MB', summary_mb),
            ('NUTS', summary_nuts),
            ('ADVI', summary_advi),
        ]:
            rows.append(self._makeRow(label, summary, fit_label))

        # --- converged subset ---
        if self.cfg.converged_subset:
            conv_mask = nutsConvergeMask(full_batch, mode=self.cfg.convergence_mode)
            if conv_mask is not None:
                n_conv = int(conv_mask.sum())
                n_total = len(conv_mask)
                logger.info(
                    '\nConverged subset (%s): %d / %d datasets',
                    self.cfg.convergence_mode, n_conv, n_total,
                )
                if 0 < n_conv < n_total:
                    conv_batch = subsetBatch(full_batch, conv_mask)
                    conv_mb = subsetProposal(proposal_mb, conv_mask)
                    conv_nuts = subsetProposal(proposal_nuts, conv_mask)
                    conv_advi_mask = advi_mask & conv_mask
                    conv_advi_batch = subsetBatch(full_batch, conv_advi_mask)
                    conv_advi = subsetProposal(proposal_advi, conv_mask[advi_mask])
                    summary_mb_conv = self.summary(conv_mb, conv_batch)
                    summary_nuts_conv = self.summary(conv_nuts, conv_batch)
                    summary_advi_conv = self.summary(conv_advi, conv_advi_batch)
                    for label, summary in [
                        ('MB', summary_mb_conv),
                        ('NUTS', summary_nuts_conv),
                        ('ADVI', summary_advi_conv),
                    ]:
                        rows.append(self._makeRow(label + '_conv', summary, fit_label))
                    conv_plot_dir = self.plot_dir / 'conv'
                    conv_plot_dir.mkdir(parents=True, exist_ok=True)
                    self.plot(
                        [conv_mb, conv_nuts],
                        [summary_mb_conv, summary_nuts_conv],
                        ['MB', 'NUTS'],
                        conv_batch,
                        plot_dir=conv_plot_dir,
                    )
                    self.plot(
                        [conv_advi],
                        [summary_advi_conv],
                        ['ADVI'],
                        conv_advi_batch,
                        plot_dir=conv_plot_dir / 'advi',
                    )

                    # --- converged + reliable LOO subset (NUTS k filter only) ---
                    k_thr = self.cfg.pareto_k_thr
                    nuts_k = summary_nuts_conv.loo_pareto_k
                    if nuts_k is not None:
                        k_mask = (nuts_k < k_thr).numpy()
                        n_k = int(k_mask.sum())
                        logger.info(
                            'Reliable LOO subset (NUTS k<%.1f): %d / %d', k_thr, n_k, n_conv
                        )
                        if 0 < n_k < n_conv:
                            k_batch = subsetBatch(conv_batch, k_mask)
                            k_mb   = subsetProposal(conv_mb,   k_mask)
                            k_nuts = subsetProposal(conv_nuts, k_mask)
                            k_advi_batch = subsetBatch(conv_advi_batch, k_mask[conv_advi_mask[conv_mask]])
                            k_advi = subsetProposal(conv_advi, k_mask[conv_advi_mask[conv_mask]])
                            summary_mb_k   = self.summary(k_mb,   k_batch)
                            summary_nuts_k = self.summary(k_nuts, k_batch)
                            summary_advi_k = self.summary(k_advi, k_advi_batch)
                            for label, summary in [
                                ('MB', summary_mb_k),
                                ('NUTS', summary_nuts_k),
                                ('ADVI', summary_advi_k),
                            ]:
                                rows.append(self._makeRow(label + '_loo', summary, fit_label))

        if self.cfg.save_tables:
            self.saveTables(rows)


# =============================================================================
def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    evaluator = Evaluator(cfg)
    # evaluator.testrun()
    evaluator.go()


if __name__ == '__main__':
    main()
