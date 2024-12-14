import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Tuple, Callable, Union
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch import distributions as D
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader
import schedulefree
from dataset import LMDataset
from models import GRU, LSTM, TransformerDecoder
from generate import dsFilename


# -------- miscellaneous
class Logger:
    def __init__(self, path: Path) -> None:
        self.trunk = path
        self.trunk.mkdir(parents=True, exist_ok=True)
        self.init("loss_train")
        self.init("loss_val")
        self.init("loss_kl")

    def init(self, losstype: str) -> None:
        fname = Path(self.trunk, f"{losstype}.csv")
        if not os.path.exists(fname):
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['iteration', 'step', 'loss'])

    def write(self, iteration: int, step: int, loss: float, losstype: str) -> None:
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


def getDataLoader(filename: Path, batch_size: int) -> DataLoader:
    ''' Load a dataset from a file, split into train and validation set and return a DataLoader. '''
    assert filename.exists(), f"File {filename} does not exist, you must generate it first using generate.py"
    ds_raw = torch.load(filename, weights_only=False)
    ds = LMDataset(**ds_raw)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def modelID(cfg: argparse.Namespace) -> str:
    ''' Return a string that identifies the model. '''
    noise = "variable" if cfg.fixed == 0 else cfg.fixed
    return f"{cfg.model}-{cfg.hidden}-{cfg.ff}-{cfg.heads}-{cfg.layers}-noise={noise}-seed={cfg.seed}-loss={cfg.loss}"


def getCheckpointPath(iteration: int) -> Path:
    ''' Get the filename for the model checkpoints. '''
    model_id = modelID(cfg)
    model_filename = f"{model_id}-{iteration:02d}.pt"
    return Path(cfg.model_folder, model_filename)


def save(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         current_iteration: int,
         current_global_step: int) -> None:
    """ Save the model and optimizer state. """
    model_filename = getCheckpointPath(current_iteration)
    os.makedirs(cfg.model_folder, exist_ok=True)
    torch.save({
        'iteration': current_iteration,
        'global_step': current_global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_filename)


def load(model: nn.Module,
         optimizer: schedulefree.AdamWScheduleFree,
         initial_iteration: int) -> Tuple[int, int]:
    """ Load the model and optimizer state from a previous run,
    returning the initial iteration and seed. """
    model_filename = getCheckpointPath(initial_iteration)
    print(f"Loading checkpoint from {model_filename}")
    state = torch.load(model_filename, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    initial_iteration = state["iteration"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    global_step = state["global_step"]
    return initial_iteration, global_step


def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=0, help="Model seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use [cuda, cpu]")
    parser.add_argument("-p", "--preload", type=int, default=0, help="Preload model from iteration #p")
    parser.add_argument("--model-folder", type=str, default="checkpoints", help="Model folder")
    parser.add_argument("--proto", action="store_true", help="prototyping: don't log anything during")

    # data
    parser.add_argument("-d", type=int, default=15, help="Number of predictors (+ bias)")
    parser.add_argument("-f", "--fixed", type=float, default=0., help='Fixed noise variance (default = 0. -> not fixed)')
    parser.add_argument("-i", "--iterations", type=int, default=500, help="Number of iterations to train")
    parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size")

    # model and loss
    parser.add_argument("-l", "--loss", type=str, default="lognormal", help="Loss function [mse, lognormal]")
    parser.add_argument("-m", "--model", type=str, default="transformer", help="Model type [gru, lstm, transformer]")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--ff", type=int, default=256, help="Feedforward dimension (transformer)")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads (transformer)")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers (transformer)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate (Adam)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon (Adam)")
    
    return parser.parse_args()


# -------- loss calculation
def logNormalLoss(means: torch.Tensor,
                  stds: torch.Tensor,
                  betas: torch.Tensor) -> torch.Tensor:
    ''' Compute the negative log density of betas (target) under the proposed normal distribution. '''
    # means (batch, n_features)
    # sigma (batch, n_features)
    # betas (batch, n_features)
    proposal = D.Normal(means, stds)
    return -proposal.log_prob(betas) # (batch, n, d)


def logInvGammaLoss(ab: torch.Tensor,
                    sigma_error: torch.Tensor) -> torch.Tensor:
    ''' Compute the negative log density of the noise std (target) under the proposed inverse gamma distribution. '''
    # ab (batch, n, 2)
    # sigma_error (batch, n, 1)
    proposal = D.inverse_gamma.InverseGamma(ab[:,:,0], ab[:,:,1])
    return -proposal.log_prob(sigma_error) # (batch, n)


def invGammaMAP(alpha_betas: torch.Tensor) -> torch.Tensor:
    ''' Compute the MAP noise variance '''
    # alpha_betas (batch, n, 2)
    alpha, beta = alpha_betas[:,:,0], alpha_betas[:,:,1]
    proposal = D.inverse_gamma.InverseGamma(alpha, beta)
    return proposal.mode # beta / (alpha + 1)


def betaLossWrapper(means: torch.Tensor,
                    sigma: torch.Tensor,
                    betas: torch.Tensor,
                    d: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the beta loss function.
    Handles the case 3D tensors (where the second dimension is the number of subjects = seq_size).
    Drop the losses for datasets that have fewer than n = noise_tol * number of features.'''
    # calculate losses for all dataset sizes and each beta
    b, n, _ = means.shape
    target = betas.unsqueeze(1).expand_as(means)
    losses = lf(means, sigma, target) # (b, n, d)

    # mask out first n_min losses per batch, then average over subject
    n_min = noise_tol * d.unsqueeze(1) # (b, 1)
    denominators = n - n_min # (b, 1)
    mask = torch.arange(n).expand(b, n) < n_min # (b, n)
    losses[mask] = 0.
    losses = losses.sum(dim=1) / denominators
    return losses # (batch, d)


def noiseLossWrapper(alpha_betas: torch.Tensor,
                     noise_std: torch.Tensor,
                     d: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the noise loss function. ''' 
    # calculate losses for all dataset sizes and each beta
    b, n, _ = alpha_betas.shape
    noise_var = torch.square(noise_std)
    target = noise_var.expand((b,n))
    losses = lf_noise(alpha_betas, target).unsqueeze(-1)
    
    # mask out first n_min losses per batch, then average over subject
    n_min = noise_tol * d.unsqueeze(-1)
    denominators = n - n_min
    mask = torch.arange(n) < n_min
    losses[mask] = 0.
    losses = losses.sum(dim=1) / denominators
    return losses # (batch, 1)


def maskLoss(losses: torch.Tensor,
             targets: torch.Tensor,
             pad_val: float = 0.,
             minimax: bool = False) -> torch.Tensor:
    ''' Ignore padding values before aggregating the losses over batches and dims'''
    # losses (batch, d)
    mask = (targets != pad_val).float()
    masked_losses = losses * mask
    if minimax:
        max_losses, _ = torch.max(masked_losses, dim=1)
        loss = max_losses.mean()
    else:
        loss = masked_losses.sum() / mask.sum()
    return loss # (,)


# -------- training and testing methods
def run(model: nn.Module,
        batch: dict,
        num_examples: int = 0,
        unpad: bool = True,
        printer: Callable = print) -> torch.Tensor:
    ''' Run a batch through the model and return the loss. '''
    depths = batch["d"]
    y = batch["y"].to(device)
    X = batch["X"].to(device)
    beta = batch["beta"].float()
    inputs = torch.cat([X, y.unsqueeze(-1)], dim=-1)
    mu, sigma, ab = model(inputs)

    # compute beta parameter loss per batch and predictor (optionally over multiple model outputs per batch)
    losses = betaLossWrapper(mu, sigma, beta, depths) # (batch, n_predictors)
    targets = beta

    # compute noise parameter loss per batch
    noise_std = batch["sigma_error"].unsqueeze(-1).float()
    losses_noise = noiseLossWrapper(ab, noise_std, depths) # (batch, 1)
    losses = torch.cat([losses, losses_noise], dim=-1)
    targets = torch.cat([targets, noise_std], dim=-1)
    
    # compute mean loss over all batches and predictors (optionally ignoring padded predictors)
    loss = maskLoss(losses, targets) if unpad else losses.mean()

    # optionally print some examples
    for i in range(num_examples):
        mask = (beta[i] != 0.)
        beta_i = beta[i, mask].detach().numpy()        
        mu_i = batch["mu_n"][i, -1, mask].detach().numpy()
        outputs_i = mu[i, -1, mask].detach().numpy()
        printer(f"\n{console_width * '-'}")
        printer(f"True       : {beta_i}")
        printer(f"Analytical : {mu_i}")
        printer(f"Predicted  : {outputs_i}")
        printer(f"{console_width * '-'}\n")
    return loss


def compare(model: nn.Module, batch: dict) -> torch.Tensor:
    ''' Compate analytical posterior with proposed posterior using KL divergence '''
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    depths = batch["d"]
    b, n, _ = X.shape
    inputs = torch.cat([X, y.unsqueeze(-1)], dim=-1)

    # get analytical posterior
    mu_a = batch["mu_n"].float().squeeze(-1)
    sigma_a = batch["Sigma_n"].float()
    sigma_a = torch.diagonal(sigma_a, dim1=-2, dim2=-1) # take diagonal for comparability
    
    # get proposed posterior
    mu_p, sigma_p, ab = model(inputs)
    # noise_var = batch["sigma_error"]
    # noise_var_hat = invGammaMAP(alpha_beta)
    # stds *= noise_var_hat.unsqueeze(-1)
    
    # Compute KL divergence only for non-padded elements
    losses = torch.zeros(b, n, device=device)
    for i in range(b):
        d = depths[i]
        mu_ai = mu_a[i, :, :d]
        mu_pi = mu_p[i, :, :d]
        sigma_ai = torch.diag_embed(sigma_a[i, :, :d].square())
        sigma_pi = torch.diag_embed(sigma_p[i, :, :d].square())
        post_a = D.multivariate_normal.MultivariateNormal(mu_ai, sigma_ai)
        post_p = D.multivariate_normal.MultivariateNormal(mu_pi, sigma_pi)
        losses[i] = D.kl.kl_divergence(post_a, post_p)
    
    # average appropriately
    n_min = noise_tol * depths
    denominators = n - n_min
    mask = torch.arange(n).expand(b, n) < n_min.unsqueeze(-1)
    losses[mask] = 0.
    losses = losses.sum(dim=1) / denominators
    return losses.mean()


def savePredictions(model: nn.Module, batch: dict, iteration_index: int, batch_index: int) -> None:
    ''' save model outputs '''
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    inputs = torch.cat([X, y.unsqueeze(-1)], dim=-1)
    mu, sigma, _ = model(inputs)
    fname = Path(pred_path, f"predictions_i={iteration_index}_b={batch_index}.pt")
    out = {
        "means": mu,
        "stds": sigma,
    }
    torch.save(out, fname)


def train(model: nn.Module,
          optimizer: schedulefree.AdamWScheduleFree,
          dataloader: DataLoader,
          writer: Union[SummaryWriter, None],
          logger: Union[Logger, None],
          iteration: int,
          step: int) -> int:
    ''' Train the model for a single iteration. '''
    model.train()
    optimizer.train()
    iterator = tqdm(dataloader, desc=f"iteration {iteration:02d} [T]")
    for batch in iterator:
        optimizer.zero_grad()
        loss = run(model, batch, unpad=True)
        loss.backward()
        optimizer.step()
        iterator.set_postfix({"loss": loss.item()})
        if writer is not None:
            writer.add_scalar("loss_train", loss.item(), step)
        if logger is not None:
            logger.write(iteration, step, loss.item(), "loss_train")
        step += 1
    return step


def validate(model: nn.Module,
             optimizer: schedulefree.AdamWScheduleFree,
             dataloader: DataLoader,
             writer: Union[SummaryWriter, None],
             logger: Union[Logger, None],
             iteration: int,
             step: int) -> int:
    ''' Validate the model for a single iteration. '''
    model.eval()
    optimizer.eval()
    iterator = tqdm(dataloader, desc=f"iteration {iteration:02d} [V]")
    with torch.no_grad():
        for j, batch in enumerate(iterator):
            # preset validation loss
            loss_val = run(model, batch, unpad=True, printer=iterator.write)
            iterator.set_postfix({"loss": loss_val.item()})
            if writer is not None:
                writer.add_scalar("loss_val", loss_val.item(), step)
            if logger is not None:
                logger.write(iteration, step, loss_val.item(), "loss_val")

            # # KL loss
            # loss_kl = compare(model, batch)
            # writer.add_scalar("loss_kl", loss_kl.item(), step)
            # logger.write(iteration, step, loss_kl.item(), "loss_kl")

            # optionally save predictions
            if iteration % 5 == 0 and not cfg.proto:
                savePredictions(model, batch, iteration, j)

            # optionally print predictions
            if iteration % 5 == 0 and j % 15 == 0:
                run(model, batch, unpad=True, printer=iterator.write, num_examples=1) # type: ignore
                
            step += 1
    return step


###############################################################################

if __name__ == "__main__":
    cfg = setup()

    # global variables
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    console_width = getConsoleWidth()
    noise_tol = 0 # minimum n for evaluation: noise_tol * d, ignore all loss values below
    noise = "variable" if cfg.fixed == 0 else cfg.fixed
    suffix = f"-noise={noise}"
    if cfg.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # model
    if cfg.model in ["gru", "lstm"]:
        if cfg.layers == 1:
            cfg.dropout = 0
            print('Setting dropout to 0 for RNN with 1 layer')
        Model = eval(cfg.model.upper())
        model = Model(num_predictors=cfg.d,
                      hidden_size=cfg.hidden,
                      n_layers=cfg.layers,
                      dropout=cfg.dropout,
                      seed=cfg.seed).to(device)
        print(f"Model: {cfg.model.upper()} with {cfg.hidden} hidden units, {cfg.layers} layer(s), {cfg.dropout} dropout")
    elif cfg.model == "transformer":
        model = TransformerDecoder(
                num_predictors=cfg.d,
                hidden_size=cfg.hidden,
                ff_size=cfg.ff,
                n_heads=cfg.heads,
                n_layers=cfg.layers,
                dropout=cfg.dropout,
                seed=cfg.seed).to(device)
        print(f"Model: Transformer with {cfg.hidden} hidden units, " + \
                f"{cfg.ff} feedforward units, {cfg.heads} heads, {cfg.layers} layer(s), " + \
                f"{cfg.dropout} dropout")
    else:
        raise ValueError(f"Model {cfg.model} not recognized.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # loss, optimizer, writer
    if cfg.loss == "mse":
        mse = nn.MSELoss(reduction='none')
        lf = lambda means, stds, targets: mse(means, targets)
    elif cfg.loss == "lognormal":
        lf = logNormalLoss
        lf_noise = logInvGammaLoss
    else:
        raise ValueError(f"Loss {cfg.loss} not recognized.")
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr, eps=cfg.eps)
    if cfg.proto:
        writer, logger = None, None
    else:
        writer = SummaryWriter(Path("runs", modelID(cfg), timestamp))
        logger = Logger(Path("losses", modelID(cfg), timestamp))
    pred_path = Path("predictions", modelID(cfg), timestamp)
    pred_path .mkdir(parents=True, exist_ok=True)
    print(f"Number of parameters: {num_params}, Loss: {cfg.loss}, Learning rate: {cfg.lr}, Epsilon: {cfg.eps}, Seed: {cfg.seed}, Device: {device}")

    # optionally preload a model
    initial_iteration, global_step, validation_step = 1, 0, 0
    if cfg.preload:
        initial_iteration, global_step = load(model, optimizer, cfg.preload)
        print(f"Preloaded model from iteration {cfg.preload}, starting at iteration {initial_iteration}.")
    else:
        print("No preloaded model found, starting from scratch.")

    # training loop
    print("Preparing validation dataset...")
    fname = Path('data', f'dataset-val{suffix}.pt')
    dataloader_val = getDataLoader(fname, 100)
    print(f"Training for {cfg.iterations + 1 - initial_iteration} iterations with 10k datasets per iteration and a batch size of {cfg.batch_size}...")
    for iteration in range(initial_iteration, cfg.iterations + 1):
        fname = dsFilename(int(1e4), iteration, suffix)
        dataloader_train = getDataLoader(fname, cfg.batch_size)
        global_step = train(model, optimizer, dataloader_train, writer, logger, iteration, global_step)
        validation_step = validate(model, optimizer, dataloader_val, writer, logger, iteration, validation_step)
        save(model, optimizer, iteration, global_step)
 
