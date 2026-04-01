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
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import (
    EvaluationSummary,
    Proposal,
    concatProposalsBatch,
    dictMean,
)
import numpy as np

from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler, runIS, runSIR
from metabeta.utils.moe import moeEstimate
from metabeta.posthoc.conformal import Calibrator
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.plot import plotComparison

logger = logging.getLogger('evaluate.py')


def setup() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--name', type=str, default='big-n-mixed', help='load configs/{name}.yaml'
    )
    parser.add_argument('--m_tag', type=str)
    parser.add_argument('--r_tag', type=str)
    parser.add_argument('--d_tag', type=str)
    parser.add_argument('--d_tag_valid', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--importance', action=argparse.BooleanOptionalAction)
    parser.add_argument('--conformal', action=argparse.BooleanOptionalAction)
    parser.add_argument('--k', type=int, default=7, help='pseudo-MoE permuted views (0=off)')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_tables', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--outdir', type=str)

    # load YAML then override with any CLI flags
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'configs' / f'{args.name}.yaml'
    with open(path, 'r') as p:
        cfg = yaml.safe_load(p)
    cfg.update(vars(args))
    if cfg.get('save_tables') is None:
        cfg['save_tables'] = True
    return argparse.Namespace(**cfg)


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
        if not hasattr(self.cfg, 'outdir'):
            self.cfg.outdir = str(Path(self.dir, '..', 'outputs', 'results'))

        # checkpoint dir
        self.run_name = runName(vars(self.cfg))
        self.ckpt_dir = Path(self.dir, '..', 'outputs', 'checkpoints', self.run_name)

        # load data and model
        self._initData()
        self._initModel()
        self._load()

        # plot dir
        self.plot_dir = None
        if self.cfg.plot:
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
        self.data_cfg_train = loadDataConfig(self.cfg.d_tag)
        assimilateConfig(self.cfg, self.data_cfg_train)

        # allow overriding validation/test data tag independently from training
        self.data_cfg_valid = loadDataConfig(self.cfg.d_tag_valid)

        # keep legacy attr name for checkpoint comparison compatibility
        self.data_cfg = self.data_cfg_train

        # get dataloaders
        self.dl_valid = self._getDataLoader('valid', batch_size=self.cfg.batch_size)
        self.dl_test = self._getDataLoader('test', batch_size=self.cfg.batch_size)

    def _getDataLoader(self, partition: str, batch_size: int | None = None) -> Dataloader:
        data_cfg = self.data_cfg_valid
        data_fname = datasetFilename(data_cfg, partition)
        data_path = Path(self.dir, '..', 'outputs', 'data', data_fname)
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
            model_cfg_path = Path(self.dir, '..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
            self.model_cfg = modelFromYaml(
                model_cfg_path,
                d_ffx=self.cfg.max_d,
                d_rfx=self.cfg.max_q,
                likelihood_family=self.cfg.likelihood_family,
            )
        self.model = Approximator(self.model_cfg).to(self.device)
        self.model.eval()

    def _load(self) -> None:
        path = Path(self.ckpt_dir, self.cfg.prefix + '.pt')
        assert path.exists(), f'checkpoint not found: {path}'
        payload = torch.load(path, map_location=self.device)

        # compare configs
        if self.data_cfg != payload['data_cfg']:
            logger.warning('data config mismatch between current and checkpoint')
        if self.model_cfg.to_dict() != payload['model_cfg']:
            logger.warning('model config mismatch between current and checkpoint')

        # load states
        self.model.load_state_dict(payload['model_state'])

        # optionally compile
        if self.cfg.compile and self.device.type != 'mps':
            self.model.compile()

    def calibrate(self) -> Calibrator:
        calibrator = Calibrator()
        proposal = self.sampleMinibatched(self.dl_valid, 'Calibration')
        full_batch = self.dl_valid.fullBatch()
        full_batch = toDevice(full_batch, 'cpu')
        if self.cfg.rescale:
            full_batch = rescaleData(full_batch)
        proposal.to('cpu')
        calibrator.calibrate(proposal, full_batch)
        calibrator.save(self.run_name)
        return calibrator

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
        proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        proposal.tpd = batch[f'{prefix}_duration'].mean().item()
        return proposal

    def _sampleBatch(self, batch: dict[str, torch.Tensor]) -> Proposal:
        """Sample a proposal from a batch (no MoE)."""
        if self.cfg.importance and not self.cfg.sir:
            return runIS(self.model, batch, self.cfg)
        elif self.cfg.sir:
            return runSIR(self.model, batch, self.cfg)
        else:
            proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
            if self.cfg.rescale:
                proposal.rescale(batch['sd_y'])
            return proposal

    def _sampleMoe(self, batch: dict[str, torch.Tensor], n_datasets_seen: int) -> list[Proposal]:
        """Sample with pseudo-MoE (B=1 per dataset), optionally applying IS."""
        B = batch['X'].shape[0]
        lf = self.cfg.likelihood_family
        proposals = []
        for i in range(B):
            single = {k: v[i : i + 1] if torch.is_tensor(v) else v for k, v in batch.items()}
            rng = np.random.default_rng(self.cfg.seed + n_datasets_seen + i)
            proposal = moeEstimate(self.model, single, self.cfg.n_samples, self.cfg.k, rng=rng)
            if self.cfg.rescale:
                proposal.rescale(single['sd_y'])
            if self.cfg.importance and not self.cfg.sir:
                data_is = rescaleData(single) if self.cfg.rescale else single
                proposal = ImportanceSampler(data_is, sir=False, likelihood_family=lf)(proposal)
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
        calibrator: Calibrator | None = None,
    ) -> EvaluationSummary:
        batch = toDevice(batch, 'cpu')
        if self.cfg.rescale:
            batch = rescaleData(batch)
        proposal.to('cpu')
        lf = self.cfg.likelihood_family
        eval_summary = getSummary(proposal, batch, calibrator=calibrator, likelihood_family=lf)
        summary_table = summaryTable(eval_summary, lf)
        logger.info(summary_table)
        return eval_summary

    def plot(
        self,
        proposals: list[Proposal],
        summaries: list[EvaluationSummary],
        labels: list[str],
        batch: dict[str, torch.Tensor],
    ) -> None:
        if not self.cfg.plot:
            return
        if self.cfg.rescale:
            batch = rescaleData(batch)
        plotComparison(summaries, proposals, labels, batch, plot_dir=self.plot_dir, show=True)

    def testrun(self) -> None:
        calibrator = self.calibrate() if self.cfg.conformal else None
        full_batch = self.dl_valid.fullBatch()
        proposal_mb = self.sampleMinibatched(self.dl_valid, 'MB')
        summary_mb = self.summary(proposal_mb, full_batch, calibrator=calibrator)
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
            'ppNLL': False,
            self._fitLabel(): True,
            'tpd': False,
            'IS_eff': True,
            'Pareto_k': False,
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

    def go(self) -> None:
        calibrator = self.calibrate() if self.cfg.conformal else None
        full_batch = self.dl_test.fullBatch()

        # MB proposal
        proposal_mb = self.sampleMinibatched(self.dl_test, 'MB')
        summary_mb = self.summary(proposal_mb, full_batch, calibrator=calibrator)

        # NUTS proposal
        proposal_nuts = self._fit2proposal(full_batch, prefix='nuts')
        summary_nuts = self.summary(proposal_nuts, full_batch)

        # ADVI proposal
        proposal_advi = self._fit2proposal(full_batch, prefix='advi')
        summary_advi = self.summary(proposal_advi, full_batch)

        self.plot(
            [proposal_mb, proposal_nuts, proposal_advi],
            [summary_mb, summary_nuts, summary_advi],
            ['MB', 'NUTS', 'ADVI'],
            full_batch,
        )

        if self.cfg.save_tables:
            fit_label = self._fitLabel()
            rows = []
            for label, summary in [
                ('MB', summary_mb),
                ('NUTS', summary_nuts),
                ('ADVI', summary_advi),
            ]:
                rows.append(
                    {
                        'method': label,
                        'R': dictMean(summary.corr),
                        'NRMSE': dictMean(summary.nrmse),
                        'ECE': dictMean(summary.ece),
                        'ppNLL': summary.mnll,
                        fit_label: summary.mfit,
                        'tpd': summary.tpd,
                        'IS_eff': summary.meff,
                        'Pareto_k': summary.mk,
                    }
                )
            self.saveTables(rows)


# =============================================================================
if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    evaluator = Evaluator(cfg)
    # evaluator.testrun()
    evaluator.go()
