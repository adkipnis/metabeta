import argparse
import yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import schedulefree

from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.config import modelFromYaml
from metabeta.utils.io import setDevice, datasetFilename


def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()
 
    # misc
    parser.add_argument('-s', '--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--reproducible', action='store_false', help='use deterministic learning trajectory (default = True)')
    parser.add_argument('--cores', type=int, default=8, help='number of processor cores to use (default = 8)')
    parser.add_argument('--device', type=str, default='cpu', help='device to use [cpu, cuda, mps], (default = mps)')
    # parser.add_argument('--save', type=int, default=10, help='save model every #p iterations (default = 10)')

    # model
    parser.add_argument('--d_tag', type=str, default='toy', help='name of data config file')
    parser.add_argument('--m_tag', type=str, default='toy', help='name of model config file')
    parser.add_argument('-l', '--load', type=int, default=0, help='load model from epoch #l')
    parser.add_argument('--n_samples', type=int, default=200, help='number of samples to draw from posterior on test set')
    parser.add_argument('--compile', action='store_true', help='compile model (default = False)')

    # training & testing
    parser.add_argument('--bs_mini', type=int, default=32, help='number of regression datasets per training minibatch (default = 32)')
    parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate (default = 1e-3)')
    parser.add_argument('--reference', action='store_false', help='do a reference run before training (default = False)')
    parser.add_argument('-e', '--max_epochs', type=int, default=10, help='maximum number of epochs to train (default = 10)')
    parser.add_argument('--test_interval', type=int, default=5, help='sample posterior every #n epochs (default = 5)')
    # parser.add_argument('--patience', type=int, default=20, help='early stopping criterium (default = 20)')

    return parser.parse_args()

# -----------------------------------------------------------------------------


        # misc setup


# =============================================================================
if __name__ == '__main__':
    # --- setup training config
 

    # loop
