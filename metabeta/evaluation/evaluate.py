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
from metabeta.plot import plotRecovery, plotCoverage, plotSBC

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

    @torch.inference_mode()
    def sample(self) -> tuple[Proposal, float]:
        # expects single batch from dl
        batch = next(iter(self.dl_test))
        batch = toDevice(batch, self.device)

        # get proposal distribution
        t0 = time.perf_counter()
        if not self.cfg.sir:
            proposal, batch = self._sampleSingle(batch)
        else:
            proposal, batch = self._sampleMulti(batch)
        t1 = time.perf_counter()
        proposal.to('cpu')
        tpd = (t1 - t0) / batch['X'].shape[0]  # time per dataset
        return proposal, tpd

    def _sampleSingle(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[Proposal, dict[str, torch.Tensor]]:
        proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)

        # unnormalize proposal and batch
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
            batch = rescaleData(batch)

        # importance weighing
        if self.cfg.importance:
            imp_sampler = ImportanceSampler(batch, sir=False)
            proposal = imp_sampler(proposal)
        return proposal, batch

    def _sampleMulti(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[Proposal, dict[str, torch.Tensor]]:
        # separate normalized and unnormalized batch
        eval_batch = batch
        if self.cfg.rescale:
            eval_batch = rescaleData(batch)
        n_sir = self.cfg.n_samples // self.cfg.sir_iter
        imp_sampler = ImportanceSampler(eval_batch, sir=True, n_sir=n_sir)
        selected = []
        n_remaining = self.cfg.n_samples

        # Sampling Importance Resampling (SIR)
        while n_remaining > 0:
            proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
            if self.cfg.rescale:
                proposal.rescale(batch['sd_y'])
            proposal = imp_sampler(proposal)
            selected.append(proposal)
            n_remaining -= proposal.n_samples
        proposal = joinProposals(selected)
        return proposal, eval_batch


# =============================================================================
if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    evaluator = Evaluator(cfg)
    proposal, tpd = evaluator.sample()
