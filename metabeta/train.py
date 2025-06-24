import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Tuple, Callable, Dict
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import schedulefree

from metabeta.models.approximators import Approximator, ApproximatorFFX#, ApproximatorMFX
from metabeta.utils import setDevice, dsFilename, getConsoleWidth, modelID, getDataLoader


# -----------------------------------------------------------------------------
# logging
class Logger:
    def __init__(self, path: Path) -> None:
        self.trunk = path
        self.trunk.mkdir(parents=True, exist_ok=True)
        self.init('loss_train')
        self.init('loss_val')
        self.tb = SummaryWriter(log_path)

    def init(self, losstype: str) -> None:
        fname = Path(self.trunk, f'{losstype}.csv')
        if not os.path.exists(fname):
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['iteration', 'step', 'loss'])

    def write(self, iteration: int,
              step: int,
              loss: float,
              losstype: str) -> None:
        self.tb.add_scalar(losstype, loss, step)
        fname = Path(self.trunk, f'{losstype}.csv')
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([iteration, step, loss])


# -----------------------------------------------------------------------------
# early stopper
class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 1e-3) -> None:
        self.patience = patience
        self.delta = delta
        self.best = float('inf')
        self.counter = 0
        self.stop = False

    def update(self, loss: float) -> None:
        diff = self.best - loss
        if diff > self.delta:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.stop = True
                print('Stopping due to impatience.')


# -----------------------------------------------------------------------------
# loading and saving
def save(model: nn.Module,
         optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
         current_iteration: int,
         current_global_step: int,
         current_validation_step: int,
         timestamp: str) -> None:
    """ Save the model and optimizer state. """
    fname = Path(model_path, f"checkpoint_i={current_iteration}.pt")
    torch.save({
        'iteration': current_iteration,
        'global_step': current_global_step,
        'validation_step': current_validation_step,
        'timestamp': timestamp,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, fname)


def load(model: nn.Module,
         optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
         initial_iteration: int) -> Tuple[int, int, int, str]:
    """ Load the model and optimizer state from a previous run,
    returning the initial iteration and seed. """
    fname = Path(model_path, f"checkpoint_i={initial_iteration}.pt")
    print(f'Loading checkpoint from {fname}')
    state = torch.load(fname, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    initial_iteration = state['iteration'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
    validation_step = state['validation_step']
    timestamp = state['timestamp']
    return initial_iteration, global_step, validation_step, timestamp


# -----------------------------------------------------------------------------
