from pathlib import Path
import numpy as np
import torch
from torch import distributions as D
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from metabeta.evaluation.predictive import (
    posteriorPredictiveSample,
    posteriorPredictiveWithinGroupSD,
    intervalCheck,
)
from metabeta.utils.plot import niceify, savePlot

