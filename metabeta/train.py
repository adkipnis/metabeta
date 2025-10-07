import os
from pathlib import Path
import csv
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import schedulefree

from metabeta.utils import setDevice, dsFilename, getConsoleWidth
from metabeta.data.dataset import getDataLoader
from metabeta.models.approximators import Approximator, ApproximatorMFX
from metabeta import plot


def setup() -> argparse.Namespace:
