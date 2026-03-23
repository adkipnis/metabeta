import yaml
import time
import logging
import argparse
import torch
from pathlib import Path

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
from metabeta.utils.evaluation import EvaluationSummary, Proposal
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import runIS, runSIR
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.plot import plotComparison

logger = logging.getLogger('evaluate.py')


def setup() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--name', type=str, default='toy', help='load configs/{name}.yaml')
    parser.add_argument('--m_tag', type=str)
    parser.add_argument('--r_tag', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--importance', action=argparse.BooleanOptionalAction)
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)

    # load YAML then override with any CLI flags
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'configs' / f'{args.name}.yaml'
    with open(path, 'r') as p:
        cfg = yaml.safe_load(p)
    cfg.update(vars(args))
    return argparse.Namespace(**cfg)


# -----------------------------------------------------------------------------
class Evaluator:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.dir = Path(__file__).resolve().parent
        setSeed(cfg.seed)
        self.device = setDevice(cfg.device)

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

    def _initData(self) -> None:
        # assimilate data config
        self.data_cfg = loadDataConfig(self.cfg.d_tag)
        assimilateConfig(self.cfg, self.data_cfg)

        # get dataloaders
        # self.dl_valid = self._getDataLoader('valid')
        self.dl_test = self._getDataLoader('test')

    def _getDataLoader(self, partition: str) -> Dataloader:
        data_fname = datasetFilename(self.data_cfg, partition)
        data_path = Path(self.dir, '..', 'outputs', 'data', data_fname)
        if partition == 'test':
            data_path = data_path.with_suffix('.fit.npz')
        assert data_path.exists(), (
            f'reintegrated fit file not found: {data_path}\n'
            'Run fit.py --reintegrate first to produce this file.'
        )
        return Dataloader(data_path, batch_size=None)

    def _initModel(self) -> None:
        """Load model architecture from config and restore checkpoint weights."""
        if hasattr(self.cfg, 'model_cfg') and isinstance(self.cfg.model_cfg, ApproximatorConfig):
            self.model_cfg = self.cfg.model_cfg
        else:
            model_cfg_path = Path(self.dir, '..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
            self.model_cfg = modelFromYaml(
                model_cfg_path, d_ffx=self.cfg.max_d, d_rfx=self.cfg.max_q
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

    def _fit2proposal(self, batch: dict[str, torch.Tensor], prefix: str) -> Proposal:
        proposed = {}
        ffx = batch[f'{prefix}_ffx']
        sigma_rfx = batch[f'{prefix}_sigma_rfx']
        sigma_eps = batch[f'{prefix}_sigma_eps'].unsqueeze(-1)
        proposed['global'] = {'samples': torch.cat([ffx, sigma_rfx, sigma_eps], dim=-1)}
        proposed['local'] = {'samples': batch[f'{prefix}_rfx']}
        proposal = Proposal(proposed)
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
        return proposal

    @torch.inference_mode()
    def sample(self, batch: dict[str, torch.Tensor]) -> Proposal:
        batch = toDevice(batch, self.device)
        t0 = time.perf_counter()
        if self.cfg.importance and not self.cfg.sir:
            proposal = runIS(self.model, batch, self.cfg)
        elif self.cfg.sir:
            proposal = runSIR(self.model, batch, self.cfg)
        else:
            proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
            if self.cfg.rescale:
                proposal.rescale(batch['sd_y'])
        t1 = time.perf_counter()
        proposal.tpd = (t1 - t0) / batch['X'].shape[0]  # time per dataset
        return proposal

    def summary(self, proposal: Proposal, batch: dict[str, torch.Tensor]) -> EvaluationSummary:
        batch = toDevice(batch, 'cpu')
        if self.cfg.rescale:
            batch = rescaleData(batch)
        proposal.to('cpu')
        eval_summary = getSummary(proposal, batch)
        summary_table = summaryTable(eval_summary)
        logger.info(summary_table)
        return eval_summary

    def plot(
        self,
        proposals: list[Proposal],
        summaries: list[EvaluationSummary],
        labels: list[str],
        batch: dict[str, torch.Tensor],
    ) -> None:
        if self.cfg.rescale:
            batch = rescaleData(batch)
        plotComparison(summaries, proposals, labels, batch, plot_dir=self.plot_dir, show=True)

    def go(self) -> None:
        batch = next(iter(self.dl_test))

        # MB proposal
        proposal_mb = self.sample(batch)
        summary_mb = self.summary(proposal_mb, batch)

        # NUTS proposal
        proposal_nuts = self._fit2proposal(batch, prefix='nuts')
        summary_nuts = self.summary(proposal_nuts, batch)

        # ADVI proposal
        proposal_advi = self._fit2proposal(batch, prefix='advi')
        summary_advi = self.summary(proposal_advi, batch)

        self.plot(
            [proposal_mb, proposal_nuts, proposal_advi],
            [summary_mb, summary_nuts, summary_advi],
            ['MB', 'NUTS', 'ADVI'],
            batch,
        )


# =============================================================================
if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    evaluator = Evaluator(cfg)
    evaluator.go()
