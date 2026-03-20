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
from metabeta.utils.evaluation import EvaluationSummary, Proposal, joinProposals
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
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

