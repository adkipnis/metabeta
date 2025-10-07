from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import torch
from metabeta.data.dataset import split, getCollater, inversePermutation

DATA_DIR = Path("real")


