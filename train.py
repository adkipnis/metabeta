import os
from pathlib import Path
from datetime import datetime
from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader
import schedulefree
from rnn import GRU
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
    ds = torch.load(filename, weights_only=False)
    ds_train, ds_val = ds.randomSplit(train_ratio, shuffle=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val


def criterionUnpadded(y_pred: torch.Tensor,
                      y_true: torch.Tensor,
                      pad_val: int = 0) -> torch.Tensor:
    ''' Compute the mean squared error between y_pred and y_true, ignoring padding values.'''
    mask = (y_true != pad_val).float()
    loss = CRITERION(y_pred, y_true)  # Shape: (batch_size, num_outputs)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()  # Avoid dividing by zero if no valid entries


def getWeightsFilePath(epoch: int):
    model_filename = f"{MODEL_BASENAME}-{epoch:02d}.pt"
    return str(Path('.') / MODEL_FOLDER / model_filename)


def save(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         current_epoch: int,
         current_global_step: int) -> None:
    """ Save the model and optimizer state. """
    model_filename = getWeightsFilePath(current_epoch)
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
        unpad: bool = False,
        num_examples: int = 0) -> torch.Tensor:
    ''' Run a batch through the model and return the loss. '''
    X_batch = batch["predictors"]
    y_batch = batch["y"]
    b_batch = batch["params"].squeeze(-1)
    lengths = batch["n"]
    depths = batch["d"]
    X_y_combined = torch.cat([X_batch, y_batch], dim=-1)
    outputs = model(X_y_combined, lengths)
    loss_fn = criterionUnpadded if unpad else CRITERION
    loss = loss_fn(outputs, b_batch.float())
    for i in range(num_examples):
        d = depths[i].item()
        outputs_i = outputs[i, :d].detach().numpy()
        b_batch_i = b_batch[i, :d].detach().numpy()
        print(f"\n{CONSOLE_WIDTH * '-'}\nExample")
        print(f"Predicted : {outputs_i}")
        print(f"True      : {b_batch_i}")
        print(f"{CONSOLE_WIDTH * '-'}\n")
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
    batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
    for batch in batch_iterator:
        optimizer.zero_grad()
        loss = run(model, batch, unpad=True)
        # optionally compute the loss coordinate-wise for each predicted variable
        if COORDINATE_WISE:
            for i in range(loss.size(1)):
                coordinate_loss = loss[:, i]
                coordinate_loss.mean().backward(retain_graph=True)
            loss = loss.mean()
        else:
            loss = loss.mean()
            loss.backward()
        writer.add_scalar("loss_train", loss.item(), step)
        batch_iterator.set_postfix({"loss": loss.item()})
        optimizer.step()
        step += 1
    return step


