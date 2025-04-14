import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Tuple, Callable, Dict
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch import distributions as D
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader
import schedulefree

from utils import dsFilename
from dataset import LMDataset, FlatDataset
from models import build


# -------- miscellaneous
class Logger:
    def __init__(self, path: Path) -> None:
        self.trunk = path
        self.trunk.mkdir(parents=True, exist_ok=True)
        self.init("loss_train")
        self.init("loss_train_noise")
        self.init("loss_val")
        self.init("loss_val_noise")
        self.init("loss_kl")

    def init(self, losstype: str) -> None:
        fname = Path(self.trunk, f"{losstype}.csv")
        if not os.path.exists(fname):
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['iteration', 'step', 'loss'])

    def write(self, iteration: int,
              step: int,
              loss: float,
              losstype: str) -> None:
        fname = Path(self.trunk, f"{losstype}.csv")
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([iteration, step, loss])


def getConsoleWidth() -> int:
    ''' Determine the width of the console for formatting purposes. '''
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        print("Could not determine console width. Defaulting to 80.")
        console_width = 80
    return console_width


def getDataLoader(filename: Path, batch_size: int, emb_type: str, permute: bool = False) -> DataLoader:
    ''' Load a dataset from a file, optionally flatten data to sequence, optionally permute features, return dataloader'''
    assert filename.exists(), f"File {filename} does not exist, you must generate it first using generate.py"
    ds_raw = torch.load(filename, weights_only=False)
    Dataset = FlatDataset if emb_type == 'sequence' else LMDataset
    ds = Dataset(**ds_raw, permute=permute)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def modelID(cfg: argparse.Namespace) -> str:
    ''' Return a string that identifies the model. '''
    return f"{cfg.post_type}-{cfg.c}-transformer-{cfg.hidden}-{cfg.ff}-{cfg.heads}-{cfg.layers}-dropout={cfg.dropout}-emb={cfg.emb_type}-loss={cfg.loss_ffx}-seed={cfg.seed}-fx={cfg.fx_type}"


def getCheckpointPath(iteration: int) -> Path:
    ''' Get the filename for the model checkpoints. '''
    model_id = modelID(cfg)
    model_filename = f"{model_id}-{iteration:02d}.pt"
    return Path("outputs", cfg.model_folder, model_filename)


def save(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         current_iteration: int,
         current_global_step: int,
         current_validation_step: int,
         timestamp: str) -> None:
    """ Save the model and optimizer state. """
    model_filename = getCheckpointPath(current_iteration)
    checkpoints_path = Path("outputs", cfg.model_folder)
    os.makedirs(checkpoints_path, exist_ok=True)
    torch.save({
        'iteration': current_iteration,
        'global_step': current_global_step,
        'validation_step': current_validation_step,
        'timestamp': timestamp,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_filename)


def load(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         initial_iteration: int) -> Tuple[int, int, int, str]:
    """ Load the model and optimizer state from a previous run,
    returning the initial iteration and seed. """
    model_filename = getCheckpointPath(initial_iteration)
    print(f"Loading checkpoint from {model_filename}")
    state = torch.load(model_filename, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    initial_iteration = state["iteration"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    global_step = state["global_step"]
    validation_step = state["validation_step"]
    timestamp = state["timestamp"]
    return initial_iteration, global_step, validation_step, timestamp


# -------- training and testing methods
def maskLoss(losses: torch.Tensor,
             targets: torch.Tensor,
             pad_val: float = 0.) -> torch.Tensor:
    ''' Ignore padding values before aggregating the losses
        over batches and dims'''
    # losses (batch, d)
    mask = (targets != pad_val).float()
    masked_losses = losses * mask
    loss = masked_losses.sum() / mask.sum()
    return loss


def kldFull(mean_a: torch.Tensor, Sigma_a: torch.Tensor,
            mean_p: torch.Tensor, Sigma_p: torch.Tensor):
    ''' get Kullback Leibler divergence between analytical and proposed
        loc/scale of the posterior for each element in the batch '''
    b = mean_p.shape[0]
    losses = torch.zeros(b, device=device)
    mask = (mean_a != 0.)
    for i in range(b):
        mask_i = mask[i] 
        mu_ai = mean_a[i, mask_i]
        mu_pi = mean_p[i, mask_i]
        Sigma_ai = Sigma_a[i, mask_i][..., mask_i]
        Sigma_pi = Sigma_p[i, mask_i][..., mask_i]
        post_a = D.MultivariateNormal(mu_ai, Sigma_ai)
        post_p = D.MultivariateNormal(mu_pi, Sigma_pi)
        losses[i] = D.kl.kl_divergence(post_a, post_p)
    return losses


def kldMarginal(mean_a: torch.Tensor, var_a: torch.Tensor,
                mean_p: torch.Tensor, var_p: torch.Tensor,
                eps: float = 1e-8):
    ''' vectorized version for the case of marginal variances '''
    mask = (var_a != 0).float()
    var_a = var_a + eps
    var_p = var_p + eps
    term1 = (mean_a - mean_p).pow(2) / var_p
    term2 = var_a / var_p
    term3 = -torch.log(var_a) + torch.log(var_p)
    kl_elements = 0.5 * (term1 + term2 + term3 - 1.) * mask
    return kl_elements.sum(dim=-1)


def compareWithAnalytical(batch: dict, outputs: torch.Tensor,
                          marginal: bool = True, save: bool = True) -> torch.Tensor:
    loc_proposed, scale_proposed = model.interpreter.getLocScale(outputs)
    var_proposed = scale_proposed.square()   

    # get analytical posterior (posterior mean vector and covariance matrix)
    ffx = batch['optimal']['ffx']
    loc_analytical = ffx['mu'].float().squeeze(-1)
    Sigma_analytical = ffx["Sigma"].float()

    # correct for noise
    sigma_error = batch["sigma_error"].float()
    Sigma_analytical = sigma_error.square().unsqueeze(-1).unsqueeze(-1) * Sigma_analytical

    # get KL divergences
    if marginal:
        var_analytical = torch.diagonal(Sigma_analytical, dim1=-2, dim2=-1)
        losses = kldMarginal(loc_analytical, var_analytical,
                             loc_proposed, var_proposed)
    else:
        Sigma_proposed = torch.diag_embed(var_proposed)
        losses = kldFull(loc_analytical, Sigma_analytical,
                             loc_proposed, Sigma_proposed)
        
    # optionally save outputs
    if save:
        fname = Path(pred_path, f"kld_i={iteration}.pt")
        torch.save(losses, fname)
    return losses


# -------- outer loops

def run(model: nn.Module,
        batch: dict,
        example_indices = [],
        printer: Callable = print,
        save: bool = False,
        ) -> Dict[str, torch.Tensor]:
    ''' Run a batch through the model and return the loss. '''
    
    # optionally cast batch to device
    if device.type != 'cpu':
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
    ffx = batch['ffx']

    if cfg.fx_type == 'ffx':
        outputs_ffx, losses_ffx = model(batch)
        loss = maskLoss(losses_ffx, ffx)
        results = {'loss': loss, 'losses_ffx': losses_ffx, 'outputs_ffx': outputs_ffx}
    else:
        outputs_ffx, outputs_rfx, losses_ffx, losses_rfx = model(batch)
        losses_joint = torch.cat([losses_ffx, losses_rfx], dim=1)
        target_blueprint = torch.cat([ffx, batch['rfx'][:,0]], dim=1)
        loss = maskLoss(losses_joint, target_blueprint)
        results = {'loss': loss, 'losses_ffx': losses_ffx, 'losses_rfx': losses_rfx,
                    'outputs_ffx': outputs_ffx, 'outputs_rfx': outputs_rfx}

    # optionally print some examples
    model.interpreter_ffx.examples(example_indices, batch, outputs_ffx, printer, console_width) # type: ignore

    # optionally save outputs
    if save:
        fname = Path(pred_path, f"predictions_i={iteration}.pt")
        torch.save(outputs_ffx, fname) # todo: include rfx
    return results


def train(model: nn.Module,
          optimizer: schedulefree.AdamWScheduleFree,
          dl: DataLoader,
          step: int) -> int:
    iterator = tqdm(dl, desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [T]")
    for batch in iterator:
        model.train()
        optimizer.train()
        optimizer.zero_grad()
        results = run(model, batch)
        loss = results['loss']
        loss.backward()
        optimizer.step()
        loss_train = loss.item()
        iterator.set_postfix_str(f"loss: {loss_train:.3f}")
        writer.add_scalar("loss_train", loss_train, step)
        logger.write(iteration, step, loss_train, "loss_train")
        step += 1
    return step

def validate(model: nn.Module,
             optimizer: schedulefree.AdamWScheduleFree,
             dl: DataLoader,
             step: int) -> int:
    iterator = tqdm(dl, desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [V]")
    if step % 5 == 0:
        example_indices = [12, 29, 42]
        printer = iterator.write
        save = True
    else:
        example_indices = []
        printer = print
        save = False
        
    with torch.no_grad():
        model.eval()
        optimizer.eval()
        for batch in iterator:
            results = run(model, batch, example_indices, printer, save)
            loss_val = results['loss'].item()
            iterator.set_postfix_str(f"loss: {loss_val:.3f}")
            writer.add_scalar("loss_val", loss_val, step)
            logger.write(iteration, step, loss_val, "loss_val")
            if cfg.kl and cfg.fx_type == 'ffx' and cfg.post_type == 'mixture' and cfg.c == 1:
                losses_kl = compareWithAnalytical(batch, results['outputs_ffx'])
                loss_kl = losses_kl.mean().item()
                writer.add_scalar("loss_kl", loss_kl, step)
                logger.write(iteration, step, loss_kl, "loss_kl")
            step += 1
    return step
            


###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use [cuda, cpu]")
    parser.add_argument("-p", "--preload", type=int, default=0, help="Preload model from iteration #p")
    parser.add_argument("--model-folder", type=str, default="checkpoints", help="Model folder")

    # data
    parser.add_argument("-t", "--fx_type", type=str, default="ffx", help="Type of dataset [ffx, mfx] (default = ffx)")
    parser.add_argument("-d", type=int, default=8, help="Number of predictors (without bias, default = 8)")
    parser.add_argument("-n", type=int, default=50, help="Maximum number of samples to draw per linear model (default = 50).")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations to train (default = 10)")
    parser.add_argument("-b", "--batch-size", type=int, default=10, help="Batch size (default = 10)")

    # model and loss
    parser.add_argument("--emb_type", type=str, default="joint", help="Embedding architecture [joint, separate, sequence] (default = joint)")
    parser.add_argument("--post_type", type=str, default="mixture", help="Posterior architecture [discrete, mixture] (default = mixture)")
    parser.add_argument("-c", type=int, default=3, help="Number of mixture components resp. grid bins (default = 3)")
    parser.add_argument("--loss_ffx", type=str, default="nll", help="Loss function [mse, nll] (default = nll)")
    parser.add_argument("--loss_rfx", type=str, default="nll", help="Loss function for rfx [mse, nll] (default = nll)")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout rate (default = 0)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension (default = 128)")
    parser.add_argument("--ff", type=int, default=256, help="Feedforward dimension (transformer, default = 256)")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads (transformer, default = 8)")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers (transformer, default = 3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam, default = 1e-3)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon (Adam, default = 1e-8)")
    parser.add_argument("--kl", action="store_false", help="additionally report KL Divergence for validation set (default = True)")

    return parser.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    # --- setup config
    cfg = setup()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    console_width = getConsoleWidth()
    device = torch.device("cuda"
                          if torch.cuda.is_available() and cfg.device == "cuda"
                          else "cpu")

    # --- set up models
    model = build(cfg.d, cfg.hidden, cfg.ff, max_n=cfg.n, 
                  emb_type=cfg.emb_type, post_type=cfg.post_type,
                  n_components=cfg.c, fx_type=cfg.fx_type).to(device)
    print(f"Model: {modelID(cfg)}")

    # --- set up optimizers
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=cfg.lr, eps=cfg.eps)
    
    # --- optionally preload a model
    initial_iteration, global_step, validation_step = 1, 1, 1
    if cfg.preload:
        initial_iteration, global_step, validation_step, timestamp = \
            load(model, optimizer, cfg.preload)
        print(f"Preloaded model from iteration {cfg.preload}, starting at iteration {initial_iteration}.")

    # --- logging
    writer = SummaryWriter(Path("outputs", "runs", modelID(cfg), timestamp))
    logger = Logger(Path("outputs", "losses", modelID(cfg), timestamp))
    pred_path = Path("outputs", "predictions", modelID(cfg), timestamp)
    pred_path.mkdir(parents=True, exist_ok=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {num_params}, FFX Loss: {cfg.loss_ffx}, RFX Loss: {cfg.loss_rfx}, Learning rate: {cfg.lr}, Device: {device}")

    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # training loop
    print("Preparing validation dataset...")
    fn = dsFilename(cfg.fx_type, 'val', cfg.d, cfg.n, 600, 0)
    dataloader_val = getDataLoader(fn, 600, cfg.emb_type)
    print(f"Training for {cfg.iterations + 1 - initial_iteration} iterations with 10k datasets per iteration and a batch size of {cfg.batch_size}...")
    for iteration in range(initial_iteration, cfg.iterations + 1):
        fn = dsFilename(cfg.fx_type, 'train', cfg.d, cfg.n, int(1e4), iteration)
        dataloader_train = getDataLoader(fn, cfg.batch_size, cfg.emb_type)
        global_step = train(model, optimizer, dataloader_train, global_step)
        validation_step = validate(model, optimizer, dataloader_val, validation_step)
        if iteration % 10 == 0:
            save(model, optimizer, iteration, global_step, validation_step, timestamp)
 
