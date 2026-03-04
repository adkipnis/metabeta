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


