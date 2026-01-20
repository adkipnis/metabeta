import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np

from metabeta.simulation import hypersample, Prior, Synthesizer, Emulator, Simulator
from metabeta.utils.io import datasetFilename
from metabeta.utils.sampling import truncLogUni


