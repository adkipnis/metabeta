import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Callable
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader
import schedulefree
from rnn import GRU, LSTM, TransformerDecoder
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
                logstds: torch.Tensor,
                y_true: torch.Tensor,
                d: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the loss function.
    Handles the case of 2D and 3D tensors (where the second dimension is the number of subjects).
    For 3D tensors, drop the losses for datasets that have fewer than n < 3 * number of features.'''
    n_dims = means.dim()
    if n_dims == 3:
        y_true = y_true.unsqueeze(1).expand_as(means)
    losses = LOSS_FN(means, logstds, y_true)
    if n_dims == 3:
        b, n, _ = means.shape
        exclude = 3 * d.unsqueeze(1)
        denominator = n - exclude
        mask = torch.arange(n).expand(b, n) < exclude
        losses[mask] = 0.
        losses = losses.sum(dim=1) / denominator
    return losses # (batch, n_features)


def mseLoss(means: torch.Tensor,
            logstds: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the mean squared error loss. '''
    return nn.functional.mse_loss(means, y_true, reduction='none')


def logNormalLoss(means: torch.Tensor,
                  logstds: torch.Tensor,
                  y_true: torch.Tensor) -> torch.Tensor:
    ''' Compute the negative log probability of y_true under the distribution with means and logstds. '''
    dist = torch.distributions.Normal(means, logstds.exp())
    return -dist.log_prob(y_true)


def getWeightsFilePath(epoch: int) -> Path:
    ''' Get the filename for the model weights. '''
    model_filename = f"{MODEL_BASENAME}-{epoch:02d}.pt"
    return Path(MODEL_FOLDER, model_filename)


def save(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         current_epoch: int,
         current_global_step: int) -> None:
    """ Save the model and optimizer state. """
    model_filename = getWeightsFilePath(current_epoch)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
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
    model_filename = getWeightsFilePath(initial_epoch)
    print(f"Loading weights from {model_filename}")
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
        printer: Callable = print) -> torch.Tensor:
    ''' Run a batch through the model and return the loss. '''
    X = batch["predictors"].to(DEVICE)
    y = batch["y"].to(DEVICE)
    targets = batch["params"].squeeze(-1).to(DEVICE)
    lengths = batch["n"].to(DEVICE)
    depths = batch["d"].to(DEVICE)
    inputs = torch.cat([X, y], dim=-1)
    means, logstds = model(inputs, lengths)
    losses = lossWrapper(means, logstds, targets, depths)
    loss = maskLoss(losses, targets) if unpad else losses.mean()

    for i in range(num_examples):
        d = depths[i].item()
        outputs_i = means[i, :d].detach().numpy()
        targets_i = targets[i, :d].detach().numpy()
        printer(f"\n{CONSOLE_WIDTH * '-'}\nExample")
        printer(f"Predicted : {outputs_i}")
        printer(f"True      : {targets_i}")
        printer(f"{CONSOLE_WIDTH * '-'}\n")
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
        if epoch % 10 == 0:
            run(model, batch, unpad=True, num_examples=2) # type: ignore
    return step

if __name__ == "__main__":
    # Global variables
    SEED = 0
    N_DRAWS = int(1e4)
    N_EPOCHS = 100
    BATCH_SIZE = 64
    MAX_PREDICTORS = 15
    HIDDEN_DIM = 128
    LR = 1e-2
    MODEL_FOLDER = "weights"
    MODEL_BASENAME = f"tf-linear-{HIDDEN_DIM}"
    PRELOAD_EPOCH = 0
    COORDINATE_WISE = False
    LOSS_FN = mseLoss
    # LOSS_FN = logNormalLoss

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    CONSOLE_WIDTH = getConsoleWidth()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer, and tensorboard writer
    # model = GRU(input_size=MAX_PREDICTORS+2,
    #             hidden_size=HIDDEN_DIM,
    #             output_size=MAX_PREDICTORS+1,
    #             seed=SEED,
    #             reuse=True).to(DEVICE)
    model = TransformerDecoder(input_size=MAX_PREDICTORS+2,
                               hidden_size=HIDDEN_DIM,
                               output_size=MAX_PREDICTORS+1,
                               seed=SEED,
                               reuse=True).to(DEVICE)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=LR, eps=1e-9)
    writer = SummaryWriter(f"runs/{MODEL_BASENAME}/{TIMESTAMP}")

    # optionally preload a model
    initial_epoch, global_step, validation_step = 1, 0, 0
    if PRELOAD_EPOCH:
        initial_epoch, global_step = load(model, optimizer, PRELOAD_EPOCH)

    # start training loop
    for epoch in range(initial_epoch, N_EPOCHS+1):
        fname = dsFilename(N_DRAWS, epoch, "fixed-sigma")
        dataloader_train, dataloader_val = getDataLoaders(fname, BATCH_SIZE)
        global_step = train(model, optimizer, dataloader_train, writer, epoch, global_step)
        validation_step = validate(model, optimizer, dataloader_val, writer, epoch, validation_step)
        save(model, optimizer, epoch, global_step)
    
