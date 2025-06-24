import os
from sys import exit
from tqdm import tqdm
import argparse
import torch
from torch import distributions as D

from metabeta.tasks import FixedEffects, MixedEffects
from metabeta.utils import dsFilename
from metabeta.markov import fitFFX, fitMFX


