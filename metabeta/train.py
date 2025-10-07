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
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--m_tag', type=str, default='r', help='Suffix for model ID (default = '')')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Model seed')
    parser.add_argument('--device', type=str, default='mps', help='Device to use [cpu, cuda, mps]')
    parser.add_argument('--cores', type=int, default=8, help='Nubmer of processor cores to use (default = 8)')

    # data
    parser.add_argument('--d_tag', type=str, default='r', help='Suffix for data ID (default = '')')
    parser.add_argument('--varied', action='store_true', help='Use data with variable d/q (default = False)')
    parser.add_argument('--semi', action='store_true', help='Use semi-synthetic data (default = True)')
    parser.add_argument('-t', '--fx_type', type=str, default='mfx', help='Type of dataset [ffx, mfx] (default = ffx)')
    parser.add_argument('-d', type=int, default=3, help='Number of fixed effects (with bias, default = 5)')
    parser.add_argument('-q', type=int, default=1, help='Number of random effects (with bias, default = 1)')
    parser.add_argument('-m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('-n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    parser.add_argument('--permute', action='store_true', help='Permute slope variables for uniform learning across heads (default = True)')

    # training
    parser.add_argument('--patience', type=int, default=25, help='Maximum number of iterations without improvement before Early Stopping (default = 5)')
    parser.add_argument('--bs-train', type=int, default=4096, help='macro batch size per training partition (default = 4096).')
    parser.add_argument('--bs-val', type=int, default=256, help='macro batch size for validation partition (default = 256).')
    parser.add_argument('--bs-mini', type=int, default=32, help='mini batch size (default = 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (Adam, default = 5e-4)')
    parser.add_argument('--standardize', action='store_false', help='Standardize inputs (default = True)')
    parser.add_argument('-p', '--preload', type=int, default=0, help='Preload model from iteration #p')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Maximum number of iterations to train (default = 10)')

    # summary network
    parser.add_argument('--sum_type', type=str, default='set-transformer', help='Summarizer architecture [set-transformer, dual-transformer] (default = set-transformer)')
    parser.add_argument('--sum_blocks', type=int, default=3, help='Number of blocks in summarizer (default = 4)')
    parser.add_argument('--sum_d', type=int, default=128, help='Model dimension (default = 128)')
    parser.add_argument('--sum_ff', type=int, default=128, help='Feedforward dimension (default = 256)')
    parser.add_argument('--sum_depth', type=int, default=1, help='Feedforward layers (default = 1)')
    parser.add_argument('--sum_out', type=int, default=64, help='Summary dimension (default = 64)')
    parser.add_argument('--sum_heads', type=int, default=8, help='Number of heads (poolformer, default = 8)')    
    parser.add_argument('--sum_dropout', type=float, default=0.01, help='Dropout rate (default = 0.01)')
    parser.add_argument('--sum_act', type=str, default='GELU', help='Activation funtction [anything implemented in torch.nn] (default = GELU)')

    # posterior network
    parser.add_argument('--post_type', type=str, default='affine', help='Normalizing flow architecture [affine, spline] (default = affine)')
    parser.add_argument('--flows', type=int, default=3, help='Number of normalizing flow blocks (default = 3)')
    parser.add_argument('--post_ff', type=int, default=128, help='Feedforward dimension (default = 128)')
    parser.add_argument('--post_depth', type=int, default=3, help='Feedforward layers (default = 3)')
    parser.add_argument('--post_dropout', type=float, default=0.01, help='Dropout rate (default = 0.01)')
    parser.add_argument('--post_act', type=str, default='ReLU', help='Activation funtction [anything implemented in torch.nn] (default = GELU)')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# logging
class Logger:
    def __init__(self, path: Path) -> None:
        self.trunk = path
        self.trunk.mkdir(parents=True, exist_ok=True)
        self.init("loss_train")
        self.init("loss_val")
        self.tb = SummaryWriter(log_path)

    def init(self, losstype: str) -> None:
        fname = Path(self.trunk, f"{losstype}.csv")
        if not os.path.exists(fname):
            with open(fname, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["iteration", "step", "loss"])

    def write(self, iteration: int, step: int, loss: float, losstype: str) -> None:
        self.tb.add_scalar(losstype, loss, step)
        fname = Path(self.trunk, f"{losstype}.csv")
        with open(fname, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([iteration, step, loss])


# -----------------------------------------------------------------------------
# early stopper
class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 1e-3) -> None:
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
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
                print("Stopping due to impatience.")


# -----------------------------------------------------------------------------
# loading and saving
def save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
    current_iteration: int,
    current_global_step: int,
    current_validation_step: int,
    timestamp: str,
) -> None:
    """Save the model and optimizer state."""
    fname = Path(model_path, f"checkpoint_i={current_iteration}.pt")
    torch.save(
        {
            "iteration": current_iteration,
            "global_step": current_global_step,
            "validation_step": current_validation_step,
            "timestamp": timestamp,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": model.stats,
        },
        fname,
    )


def load(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
    initial_iteration: int,
) -> tuple[int, int, int, str]:
    """Load the model and optimizer state from a previous run,
    returning the initial iteration and seed."""
    fname = Path(model_path, f"checkpoint_i={initial_iteration}.pt")
    print(f"Loading checkpoint from {fname}")
    state = torch.load(fname, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.stats = state["stats"]
    initial_iteration = state["iteration"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    global_step = state["global_step"]
    validation_step = state["validation_step"]
    timestamp = state["timestamp"]
    return initial_iteration, global_step, validation_step, timestamp


# -----------------------------------------------------------------------------
# the bread and butter


def run(
    model: ApproximatorMFX,
    batch: dict[str, torch.Tensor],
    sample: bool = False,
) -> dict:
    targets, names, moments = {}, {}, {}
    results = model(batch, sample=sample, s=100)
    targets["global"] = model.targets(batch, local=False)
    targets["local"] = model.targets(batch, local=True)
    names["global"] = model.names(batch, local=False)
    names["local"] = model.names(batch, local=True)
    if sample:
        moments["global"] = model.moments(results["proposed"]["global"])
        moments["local"] = model.moments(results["proposed"]["local"])
    out = {
        "loss": results["loss"],
        "proposed": results["proposed"],
        "targets": targets,
        "names": names,
        "moments": moments,
    }
    return out


def train(
    model: Approximator,
    optimizer: schedulefree.AdamWScheduleFree,
    dl: DataLoader,
    step: int,
) -> int:
    iterator = tqdm(dl, desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [T]")
    running_sum = 0.0
    model.train()
    optimizer.train()
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        results = model(batch, sample=False)
        loss = results["loss"].mean()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        running_sum += loss.item()
        loss_train = running_sum / (i + 1)
        iterator.set_postfix_str(f"loss: {loss_train:.3f}")
        logger.write(iteration, step, loss_train, "loss_train")
        step += 1
    return step


def validate(model: ApproximatorMFX, dl: DataLoader, step: int) -> int:
