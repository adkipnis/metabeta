import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Callable
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader
import schedulefree
from models import GRU, LSTM, TransformerDecoder
from generate import dsFilename


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


def getDataLoaders(filename: Path, batch_size: int, train_ratio: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    ''' Load a dataset from a file, split into train and validation set and return a DataLoader. '''
    assert filename.exists(), f"File {filename} does not exist, you must generate it first using generate.py"
    ds = torch.load(filename, weights_only=False)
    ds_train, ds_val = ds.randomSplit(train_ratio, shuffle=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val


def maskLoss(losses: torch.Tensor,
             y_true: torch.Tensor,
             pad_val: int = 0) -> torch.Tensor:
    ''' Compute the mean squared error between y_pred and y_true, ignoring padding values.'''
    mask = (y_true != pad_val).float()
    masked_loss = losses * mask
    return masked_loss.sum() / mask.sum() # Average over non-padded values


def lossWrapper(means: torch.Tensor,
                sigma: torch.Tensor,
                target: torch.Tensor,
                d: torch.Tensor,
                lengths: torch.Tensor,
                last: bool = False) -> torch.Tensor:
    ''' Wrapper for the loss function.
    Handles the case of 2D and 3D tensors (where the second dimension is the number of subjects = seq_size).
    For 3D tensors, drop the losses for datasets that have fewer than n < 3 * number of features.'''
    n_dims = means.dim()
    if n_dims == 3:
        target = target.unsqueeze(1).expand_as(means)
    losses = lf(means, sigma, target)
    if n_dims == 3:
        b, n, _ = means.shape
        if last: # only return loss for last legal output
            losses = losses[torch.arange(b), lengths-1]
        else:
            n_min = 3 * d.unsqueeze(1)
            denominators = n - n_min
            mask = torch.arange(n).expand(b, n) < n_min
            losses[mask] = 0.
            losses = losses.sum(dim=1) / denominators
    return losses # (batch, n_features)


def mseLoss(means: torch.Tensor,
            sigma: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the mean squared error loss. '''
    return nn.functional.mse_loss(means, target, reduction='none')


def logNormalLoss(means: torch.Tensor,
                  sigma: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    ''' Compute the negative log probability of target under the proposal distribution. '''
    # means (batch, n_features)
    # sigma (batch, n_features) or (batch, n_features, n_features)
    # target (batch, n_features)
    if means.shape == sigma.shape:
        dist = torch.distributions.Normal(means, sigma)
    else:
        dist = torch.distributions.MultivariateNormal(means, covariance_matrix=sigma)
    return -dist.log_prob(target) # (batch, n_features)


def modelID(cfg: argparse.Namespace) -> str:
    ''' Return a string that identifies the model. '''
    return f"{cfg.model}-{cfg.hidden_dim}-{cfg.n_layers}-{cfg.seed}"


def getCheckpointPath(epoch: int) -> Path:
    ''' Get the filename for the model checkpoints. '''
    model_id = modelID(cfg)
    model_filename = f"{model_id}-{epoch:02d}.pt"
    return Path(cfg.model_folder, model_filename)


def save(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         current_epoch: int,
         current_global_step: int) -> None:
    """ Save the model and optimizer state. """
    model_filename = getCheckpointPath(current_epoch)
    os.makedirs(cfg.model_folder, exist_ok=True)
    torch.save({
        'epoch': current_epoch,
        'global_step': current_global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_filename)


def load(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         initial_epoch: int) -> Tuple[int, int]:
    """ Load the model and optimizer state from a previous run,
    returning the initial epoch and seed. """
    model_filename = getCheckpointPath(initial_epoch)
    print(f"Loading checkpoint from {model_filename}")
    state = torch.load(model_filename, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    initial_epoch = state["epoch"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    global_step = state["global_step"]
    return initial_epoch, global_step


def run(model: nn.Module,
        batch: dict,
        num_examples: int = 0,
        unpad: bool = True,
        printer: Callable = print,
        last: bool = False) -> torch.Tensor:
    ''' Run a batch through the model and return the loss. '''
    X = batch["predictors"].to(device)
    y = batch["y"].to(device)
    targets = batch["params"].squeeze(-1).to(device)
    lengths = batch["n"].to(device)
    depths = batch["d"].to(device)
    inputs = torch.cat([X, y], dim=-1)
    means, logstds = model(inputs, lengths)

    # compute loss per batch and predictor (optionally over multiple model outputs per batch)
    losses = lossWrapper(means, logstds, targets, depths, lengths, last) # (batch, d)

    # compute mean loss over all batches and predictors (optionally ignoring padded predictors)
    loss = maskLoss(losses, targets) if unpad else losses.mean()

    # optionally print some examples
    for i in range(num_examples):
        d = depths[i].item()
        n = lengths[i].item()
        targets_i = targets[i, :d].detach().numpy()
        if model.last:
            outputs_i = means[i, :d].detach().numpy()
        else:
            outputs_i = means[i, n-1, :d].detach().numpy()
        printer(f"\n{console_width * '-'}")
        printer(f"Predicted : {outputs_i}")
        printer(f"True      : {targets_i}")
        printer(f"{console_width * '-'}\n")
    return loss


def train(model: nn.Module,
          optimizer: schedulefree.AdamWScheduleFree,
          dataloader: DataLoader,
          writer: SummaryWriter,
          epoch: int,
          step: int) -> int:
    ''' Train the model for a single epoch. '''
    model.train()
    optimizer.train()
    iterator = tqdm(dataloader, desc=f"Epoch {epoch:02d} [T]")
    for batch in iterator:
        optimizer.zero_grad()
        loss = run(model, batch, unpad=True)
        loss.backward()
        writer.add_scalar("loss_train", loss.item(), step)
        iterator.set_postfix({"loss": loss.item()})
        optimizer.step()
        step += 1
    return step


def validate(model: nn.Module,
             optimizer: schedulefree.AdamWScheduleFree,
             dataloader: DataLoader,
             writer: SummaryWriter,
             epoch: int,
             step: int) -> int:
    ''' Validate the model for a single epoch. '''
    model.eval()
    optimizer.eval()
    iterator = tqdm(dataloader, desc=f"Epoch {epoch:02d} [V]")
    with torch.no_grad():
        for batch in iterator:
            val_loss = run(model, batch, unpad=True, printer=iterator.write)
            writer.add_scalar("loss_val", val_loss.item(), step)
            iterator.set_postfix({"loss": val_loss.item()})
            step += 1
        if epoch % 5 == 0:
            run(model, batch, unpad=True, num_examples=2) # type: ignore
    return step

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=0, help="Model seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("-p", "--preload", type=int, default=0, help="Preload model from epoch")
    parser.add_argument("--model-folder", type=str, default="checkpoints", help="Model folder")

    # data
    parser.add_argument("--n-draws", type=int, default=int(1e4), help="Number of datasets per epoch")
    parser.add_argument("--d", type=int, default=19, help="Number of predictors")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--last", dest="last", action="store_true", help="Use only last output for loss")

    # model and loss
    parser.add_argument("-m", "--model", type=str, default="transformer", help="Model type (gru, lstm, transformer)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--ff-dim", type=int, default=128, help="Feedforward dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of heads in transformer")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of layers in transformer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for Adam")
    parser.add_argument("--loss", type=str, default="lognormal", help="Loss function (mse, lognormal)")

    return parser.parse_args()


if __name__ == "__main__":
    cfg = setup()

    # global variables
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    console_width = getConsoleWidth()
    if cfg.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # model
    if cfg.model in ["gru", "lstm"]:
        if cfg.n_layers == 1:
            cfg.dropout = 0
            print('Setting dropout to 0 for RNN with 1 layer')
        Model = eval(cfg.model.upper())
        model = Model(num_predictors=cfg.d,
                      hidden_size=cfg.hidden_dim,
                      n_layers=cfg.n_layers,
                      dropout=cfg.dropout,
                      seed=cfg.seed,
                      last=cfg.last).to(device)
        print(f"Model: {cfg.model.upper()} with {cfg.hidden_dim} hidden units, {cfg.n_layers} layer(s), {cfg.dropout} dropout")
    elif cfg.model == "transformer":
        model = TransformerDecoder(
                num_predictors=cfg.d,
                hidden_size=cfg.hidden_dim,
                ff_size=cfg.ff_dim,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
                seed=cfg.seed,
                last=cfg.last).to(device)
        print(f"Model: Transformer with {cfg.hidden_dim} hidden units, " + \
                f"{cfg.ff_dim} feedforward units, {cfg.n_heads} heads, {cfg.n_layers} layer(s), " + \
                f"{cfg.dropout} dropout")
    else:
        raise ValueError(f"Model {cfg.model} not recognized.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # loss, optimizer, writer
    if cfg.loss == "mse":
        lf = mseLoss
    elif cfg.loss == "lognormal":
        lf = logNormalLoss
    else:
        raise ValueError(f"Loss {cfg.loss} not recognized.")
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr, eps=cfg.eps)
    writer = SummaryWriter(Path("runs", modelID(cfg), timestamp))
    print(f"Number of parameters: {num_params}, Loss: {cfg.loss}, Learning rate: {cfg.lr}, Epsilon: {cfg.eps}, Seed: {cfg.seed}, Device: {device}")

    # optionally preload a model
    initial_epoch, global_step, validation_step = 1, 0, 0
    if cfg.preload:
        initial_epoch, global_step = load(model, optimizer, cfg.preload)
        print(f"Preloaded model from epoch {cfg.preload}, starting at epoch {initial_epoch}.")
    else:
        print("No preloaded model found, starting from scratch.")

    # start training loop
    sfx = "(using only last model output for loss)" if cfg.last else ""
    print(f"Training for {cfg.epochs + 1 - initial_epoch} epochs with {cfg.n_draws} datasets per epoch... {sfx}")
    for epoch in range(initial_epoch, cfg.epochs + 1):
        fname = dsFilename(cfg.n_draws, epoch, "fixed-sigma")
        dataloader_train, dataloader_val = getDataLoaders(fname, cfg.batch_size)
        global_step = train(model, optimizer, dataloader_train, writer, epoch, global_step)
        validation_step = validate(model, optimizer, dataloader_val, writer, epoch, validation_step)
        save(model, optimizer, epoch, global_step)
 
