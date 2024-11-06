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


def maskLoss(losses: torch.Tensor,
             y_true: torch.Tensor,
             pad_val: int = 0) -> torch.Tensor:
    ''' Compute the mean squared error between y_pred and y_true, ignoring padding values.'''
    mask = (y_true != pad_val).float()
    masked_loss = losses * mask
    return masked_loss.sum() / mask.sum() # Average over non-padded values


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


def validate(model: nn.Module,
             optimizer: schedulefree.AdamWScheduleFree,
             dataloader: DataLoader,
             writer: SummaryWriter,
             epoch: int,
             step: int) -> int:
    ''' Validate the model for a single epoch. '''
    model.eval()
    optimizer.eval()
    with torch.no_grad():
        for batch in dataloader:
            val_loss = run(model, batch, unpad=True)
            writer.add_scalar("loss_val", val_loss.item(), step)
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
    MODEL_BASENAME = f"rnn-linear-{HIDDEN_DIM}"
    PRELOAD_EPOCH = 0
    COORDINATE_WISE = False
    CRITERION = nn.MSELoss(reduction='none')
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    CONSOLE_WIDTH = getConsoleWidth()

    # Initialize model, optimizer, and tensorboard writer
    model = GRU(input_size=MAX_PREDICTORS+2,
                hidden_size=HIDDEN_DIM,
                output_size=MAX_PREDICTORS+1,
                seed=SEED)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=LR, eps=1e-9)
    writer = SummaryWriter(f"runs/{MODEL_BASENAME}/{TIMESTAMP}")

    # optionally preload a model
    initial_epoch, global_step, validation_step = 1, 0, 0
    if PRELOAD_EPOCH:
        initial_epoch, global_step = load(model, optimizer, PRELOAD_EPOCH)

    # start training loop
    for epoch in range(initial_epoch, N_EPOCHS+1):
        fname = dsFilename(N_DRAWS, epoch)
        dataloader_train, dataloader_val = getDataLoaders(fname, BATCH_SIZE)
        global_step = train(model, optimizer, dataloader_train, writer, epoch, global_step)
        validation_step = validate(model, optimizer, dataloader_val, writer, epoch, validation_step)
        save(model, optimizer, epoch, global_step)
    
