import argparse
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import time
from collections.abc import Iterable
import torch
from torch import distributions as D
import numpy as np
from scipy.stats import pearsonr
from metabeta.data.dataset import getDataLoader
from metabeta.utils import dsFilename, getConsoleWidth
from metabeta.models.approximators import ApproximatorMFX
from metabeta.evaluation.importance import ImportanceLocal, ImportanceGlobal
from metabeta.evaluation.coverage import getCoverage, plotCalibration, coverageError
from metabeta.evaluation.sbc import getRanks, plotSBC, plotECDF, getWasserstein
from metabeta.evaluation.pp import (
    posteriorPredictiveSample,
    plotPosteriorPredictive,
    weightSubset,
)
from metabeta import plot

CI = [50, 68, 80, 90, 95]
plt.rcParams["figure.dpi"] = 300

###############################################################################

