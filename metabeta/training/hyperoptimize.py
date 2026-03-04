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
