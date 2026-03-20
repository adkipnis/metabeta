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
