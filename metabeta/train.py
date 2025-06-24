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
# the bread and butter
def run(model: Approximator,
        batch: dict,
        example_indices = [],
        printer: Callable = print,
        sample: bool = False,
        device: str = 'cpu',
        ) -> Dict[str, torch.Tensor]:
    ''' Run a batch through the model and return the loss. '''
    # optionally cast batch to device
    if device != 'cpu':
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
    # forward pass
    loss, proposed, summary = model(batch, sample=sample, local=cfg.local)
    # optionally print some examples
    model.examples(example_indices, batch, proposed, printer, console_width)
    return {'loss': loss, 'proposed': proposed, 'summary': summary}


def train(model: Approximator,
          optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
          dl: DataLoader,
          step: int) -> int:
    iterator = tqdm(dl, desc=f'iteration {iteration:02d}/{cfg.iterations:02d} [T]')
    running_sum = 0.
    model.train()
    optimizer.train()
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        results = run(model, batch, sample=False, device=device.type)
        loss = results['loss'].mean()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        running_sum += loss.item()
        loss_train = running_sum / (i+1)
        iterator.set_postfix_str(f'loss: {loss_train:.3f}')
        logger.write(iteration, step, loss_train, 'loss_train')
        step += 1
    model.eval()
    optimizer.eval()
    return step


def validate(model: Approximator, dl: DataLoader, step: int) -> int:
    iterator = tqdm(dl, desc=f'iteration {iteration:02d}/{cfg.iterations:02d} [V]')
    if step % 10 == 0:
        example_indices = range(3)
        printer = iterator.write
        sample = True
    else:
        example_indices = []
        printer = print
        sample = False
    with torch.no_grad():
        model = model.to('cpu')
        for batch in iterator:
            results = run(model, batch, example_indices, printer, sample=sample, device='cpu')
            loss_val = results['loss'].mean().item()
            iterator.set_postfix_str(f'loss: {loss_val:.3f}')
            logger.write(iteration, step, loss_val, 'loss_val')
            stopper.update(loss_val)
            step += 1
        model = model.to(device)
    return step


###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--device", type=str, default="mps", help="Device to use [cpu, cuda, mps]")
    parser.add_argument("--cores", type=int, default=4, help="Nubmer of processor cores to use (default = 4)")
    parser.add_argument("-p", "--preload", type=int, default=0, help="Preload model from iteration #p")

    # data & training
    parser.add_argument("-t", "--fx_type", type=str, default="ffx", help="Type of dataset [ffx, mfx] (default = ffx)")
    parser.add_argument("-d", type=int, default=1, help="Number of fixed effects (without bias, default = 1)")
    parser.add_argument("-m", type=int, default=0, help="Maximum number of groups (default = 0).")
    parser.add_argument("-n", type=int, default=50, help="Maximum number of samples per group (default = 50).")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Maximum number of iterations to train (default = 50)")
    parser.add_argument("--permute", action='store_false', help="Permute slope variables for uniform learning across heads (default = True)")
    parser.add_argument("--patience", type=int, default=5, help="Maximum number of iterations without improvement before Early Stopping (default = 5)")
    parser.add_argument("--local", action='store_false', help="Infer local variables (default = True)")
    parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size (default = 50)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (Adam, default = 5e-4)")

    # model
    parser.add_argument("--emb_type", type=str, default="joint", help="Embedding architecture [joint, separate, sequence] (default = joint)")
    parser.add_argument("--sum_type", type=str, default="poolformer", help="Summarizer architecture [deepset, poolformer] (default = deepset)")
    parser.add_argument("--post_type", type=str, default="flow-affine", help="Posterior architecture [point, discrete, mixture, flow-affine, flow-spline, flow-matching] (default = flow)")
    parser.add_argument("--bins", type=int, default=500, help="Number of bins in discrete posterior (default = 500)")
    parser.add_argument("--components", type=int, default=3, help="Number of mixture components (default = 3)")
    parser.add_argument("--flows", type=int, default=3, help="Number of normalizing flow blocks (default = 3)")
    parser.add_argument("--dropout", type=float, default=0.01, help="Dropout rate (default = 0.01)")
    parser.add_argument("--act", type=str, default="Mish", help="Activation funtction [anything implemented in torch.nn] (default = GELU)")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension (default = 64)")
    parser.add_argument("--ff", type=int, default=128, help="Feedforward dimension (default = 128)")
    parser.add_argument("--out", type=int, default=32, help="Summary dimension (default = 32)")
    parser.add_argument("--blocks", type=int, default=3, help="Number of blocks in summarizer (default = 3)")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads (poolformer, default = 8)")
    
    return parser.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    # --- setup config
    cfg = setup()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    console_width = getConsoleWidth()
    device = setDevice(cfg.device)
    torch.set_num_threads(cfg.cores)

    # --- set up model
    Approx = ApproximatorMFX if cfg.fx_type == "mfx" else ApproximatorFFX
    model = Approx.build(cfg.d, cfg.hidden, cfg.ff, cfg.out,
                         cfg.dropout, cfg.act,
                         cfg.heads, cfg.blocks,
                         cfg.emb_type, cfg.sum_type, cfg.post_type,
                         bins=cfg.bins, components=cfg.components, flows=cfg.flows,
                         max_m=cfg.m,
                         ).to(device)
    print(f'{"-"*console_width}\nmodel: {modelID(cfg)}')

    # --- set up optimizer
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr)

    # --- optionally preload a model
    model_path = Path('outputs', 'checkpoints', modelID(cfg))
    model_path.mkdir(parents=True, exist_ok=True)
    initial_iteration, global_step, validation_step = 1, 1, 1
    if cfg.preload:
        initial_iteration, global_step, validation_step, timestamp = \
            load(model, optimizer, cfg.preload)
        print(f'preloaded model from iteration {cfg.preload}, starting at iteration {initial_iteration}...')

    # --- logging and stopping
    log_path = Path('outputs', 'losses', modelID(cfg), timestamp)
    logger = Logger(log_path)
    stopper = EarlyStopping(patience=cfg.patience)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'parameters: {num_params}\nLearning rate: {cfg.lr}\nDevice: {device}')

    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # training loop
    print(f'features (max): {cfg.d}\nobservations (max): {cfg.n}')
    fn = dsFilename(cfg.fx_type, 'val', cfg.d, cfg.m, cfg.n, 500, 0)
    dl_val = getDataLoader(fn, 500, permute=cfg.permute)

    print(f'iterations: {cfg.iterations + 1 - initial_iteration}\npatience: {cfg.patience}\nbatches per iteration: 200\ndatasets per batch: {cfg.batch_size}\n{"-"*console_width}')
    for iteration in range(initial_iteration, cfg.iterations + 1):
        fn = dsFilename(cfg.fx_type, 'train', cfg.d, cfg.m, cfg.n, int(1e4), iteration)
        dl_train = getDataLoader(fn, cfg.batch_size, permute=cfg.permute)
        global_step = train(model, optimizer, dl_train, global_step)
        validation_step = validate(model, dl_val, validation_step)
        if iteration % 5 == 0 or stopper.stop:
            save(model, optimizer, iteration, global_step, validation_step, timestamp)
        if stopper.stop:
            break
