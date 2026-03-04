from typing import Callable
import yaml
import argparse
import logging
from pathlib import Path
import copy
import optuna

from metabeta.training.train import Trainer
from metabeta.utils.config import modelFromYaml, ApproximatorConfig, SummarizerConfig, PosteriorConfig
from metabeta.utils.evaluation import dictMean
from metabeta.utils.logger import setupLogging


logger = logging.getLogger('hyperoptimize.py')


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--name', type=str, default='hyper', help='load configs/{name}.yaml')
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--sampler', type=str, default='tpe', choices=['tpe', 'random'])
    parser.add_argument('--pruner', type=str, default='median', choices=['median', 'none'])

    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'configs' / f'{args.name}.yaml'
    with open(path, 'r') as p:
        cfg = yaml.safe_load(p)
    cfg.update(vars(args))
    return argparse.Namespace(**cfg)


class HyperOptimizer:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.dir = Path(__file__).resolve().parent

        # data config
        data_cfg_path = Path(self.dir, '..', 'simulation', 'configs', f'{self.cfg.d_tag}.yaml')
        assert data_cfg_path.exists(), f'config file {data_cfg_path} does not exist'
        with open(data_cfg_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            d_ffx, d_rfx = data_cfg['max_d'], data_cfg['max_q']

        # base model config
        model_cfg_path = Path(self.dir, '..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
        self.model_cfg = modelFromYaml(model_cfg_path, d_ffx=d_ffx, d_rfx=d_rfx).to_dict()

        # optuna study
        self.study = optuna.create_study(
            study_name=self.cfg.name,
            directions=['minimize', 'minimize'],
            sampler=optuna.samplers.TPESampler(),
            load_if_exists=True,
        )

        # output dir
        self.out_dir = Path(self.dir, '..', 'outputs', 'optuna', self.cfg.name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def suggest(self, trial: optuna.Trial) -> ApproximatorConfig:
