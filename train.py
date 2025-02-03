import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Tuple, Callable, Union, Dict
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch import distributions as D
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader
import schedulefree
from utils import dsFilename, dsFilenameVal, getAlpha
from dataset import LMDataset
from models import TransformerDecoder


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
    ds = LMDataset(**ds_raw, permute=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def modelID(cfg: argparse.Namespace) -> str:
    ''' Return a string that identifies the model. '''
    noise = "variable" if cfg.fixed == 0 else cfg.fixed
    return f"transformer-{cfg.hidden}-{cfg.ff}-{cfg.heads}-{cfg.layers}-dropout={cfg.dropout}-noise={noise}-seed={cfg.seed}-loss={cfg.loss}"


def getCheckpointPath(iteration: int) -> Path:
    ''' Get the filename for the model checkpoints. '''
    model_id = modelID(cfg)
    model_filename = f"{model_id}-{iteration:02d}.pt"
    return Path(cfg.model_folder, model_filename)


def save(models: Tuple[nn.Module, nn.Module],
         optimizers: Tuple[schedulefree.AdamWScheduleFree, schedulefree.AdamWScheduleFree],
         current_iteration: int,
         current_global_step: int,
         current_validation_step: int,
         timestamp: str) -> None:
    """ Save the model and optimizer state. """
    model_filename = getCheckpointPath(current_iteration)
    os.makedirs(cfg.model_folder, exist_ok=True)
    torch.save({
        'iteration': current_iteration,
        'global_step': current_global_step,
        'validation_step': current_validation_step,
        'timestamp': timestamp,
        'model_state_dict': models[0].state_dict(),
        'model_state_dict_noise': models[1].state_dict(),
        'optimizer_state_dict': optimizers[0].state_dict(),
        'optimizer_state_dict_noise': optimizers[1].state_dict(),
    }, model_filename)


def load(models: Tuple[nn.Module, nn.Module],
         optimizers: Tuple[schedulefree.AdamWScheduleFree, schedulefree.AdamWScheduleFree],
         initial_iteration: int) -> Tuple[int, int, int, str]:
    """ Load the model and optimizer state from a previous run,
    returning the initial iteration and seed. """
    model_filename = getCheckpointPath(initial_iteration)
    print(f"Loading checkpoint from {model_filename}")
    state = torch.load(model_filename, weights_only=False)
    models[0].load_state_dict(state["model_state_dict"])
    models[1].load_state_dict(state["model_state_dict_noise"])
    initial_iteration = state["iteration"] + 1
    optimizers[0].load_state_dict(state["optimizer_state_dict"])
    optimizers[1].load_state_dict(state["optimizer_state_dict_noise"])
    global_step = state["global_step"]
    validation_step = state["validation_step"]
    timestamp = state["timestamp"]
    return initial_iteration, global_step, validation_step, timestamp


# -------- loss helpers
def averageOverN(losses: torch.Tensor, n: int, b: int, depths: torch.Tensor, n_min: int = 0, weigh: bool = False) -> torch.Tensor:
    ''' mask out first n_min losses per batch, then calculate weighted average over n with higher emphasis on later n'''
    # losses (b, n, d)
    if cfg.tol == 0:
        return losses.mean(dim=1)
    if n_min == 0:
        n_min = cfg.tol * depths.unsqueeze(1) # (b, 1)
    denominators = n - n_min # (b, 1)
    mask = torch.arange(n).expand(b, n) < n_min # (b, n)
    losses[mask] = 0.
    if weigh:
        weights = torch.arange(0, 1, 1/n) + 1/n
        weights = weights.sqrt().unsqueeze(0).unsqueeze(-1)
        losses = losses * weights
    losses = losses.sum(dim=1) / denominators
    return losses # (batch, d)


def maskLoss(losses: torch.Tensor,
             targets: torch.Tensor,
             pad_val: float = 0.) -> torch.Tensor:
    ''' Ignore padding values before aggregating the losses over batches and dims'''
    # losses (batch, d)
    mask = (targets != pad_val).float()
    masked_losses = losses * mask
    loss = masked_losses.sum() / mask.sum()
    return loss # (,)


def moments2ig(loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' reparameterize IG-distribution (from loc and scale to alpha and beta) '''
    eps = torch.finfo(loc.dtype).eps
    alpha = 2. + torch.div(loc.square(), scale.square().clamp(min=eps))
    beta = torch.mul(loc, alpha - 1.)
    return alpha, beta


# -------- fixed effects loss
mse = nn.MSELoss(reduction='none')

def ffxMSE(loc: torch.Tensor,
           scale: torch.Tensor,
           ffx: torch.Tensor) -> torch.Tensor:
    return mse(loc, ffx)


def ffxLogProb(loc: torch.Tensor,
               scale: torch.Tensor,
               ffx: torch.Tensor) -> torch.Tensor:
    ''' Compute the negative log density of betas (target) under the proposed normal distribution. '''
    # loc (batch, n_features)
    # scale (batch, n_features)
    # ffx (batch, n_features)
    proposal = D.Normal(loc, scale)
    return -proposal.log_prob(ffx) # (batch, n, d)


def ffxLossWrapper(loc: torch.Tensor,
                   scale: torch.Tensor,
                   ffx: torch.Tensor,
                   depths: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the beta loss function.
    Handles the case 3D tensors (where the second dimension is the number of subjects = seq_size).
    Drop the losses for datasets that have fewer than n = noise_tol * number of features.'''
    # calculate losses for all dataset sizes and each beta
    b, n, _ = loc.shape
    target = ffx.unsqueeze(1).expand_as(loc)
    losses = lf_ffx(loc, scale, target) # (b, n, d)
    return averageOverN(losses, n, b, depths)


# -------- random effects loss
def rfxMSE(loc: torch.Tensor,
           scale: torch.Tensor,
           rfx_scale: torch.Tensor,
           rfx: torch.Tensor) -> torch.Tensor:
    return mse(scale.log(), rfx_scale.log())


def rfxLogProb(loc: torch.Tensor,
               scale: torch.Tensor,
               rfx_scale: torch.Tensor,
               rfx: torch.Tensor) -> torch.Tensor:
    # define a q-dimensional IG-distribution based on rfx_params
    alpha, beta = loc, scale # alternatively, but even less stable: moments2ig(loc, scale)
    proposal = D.inverse_gamma.InverseGamma(alpha, beta)
    return -proposal.log_prob(rfx_scale.square()) # (batch, n)


def rfxLossWrapper(rfx_params: torch.Tensor,
                   rfx_scale: torch.Tensor,
                   rfx: torch.Tensor,
                   depths: torch.Tensor) -> torch.Tensor:
    b, n, _, _ = rfx_params.shape
    loc, scale = rfx_params[...,0], rfx_params[...,1]
    rfx_scale += (rfx_scale == 0.).float() # set entries with std = 0 to 1
    # if stds_target.shape != rfx_params.shape:
        # stds_target = stds_target.unsqueeze(1).expand_as(rfx_params[..., 0])
    losses = lf_rfx(loc, scale, rfx_scale, rfx) # (b, n, d)
    return averageOverN(losses, n, b, depths, n_min=1)


# -------- noise parameter loss
def noiseMSE(scale: torch.Tensor,
             noise_scale: torch.Tensor) -> torch.Tensor:
    scale = scale.squeeze(-1)
    return mse(scale.log(), noise_scale.log())


def noiseLogProb(beta: torch.Tensor,
                 noise_scale: torch.Tensor) -> torch.Tensor:
    ''' Compute the negative log density of the noise std (target) under the proposed inverse gamma distribution. '''
    # beta, noise_scale (batch, n)
    beta = beta.squeeze(-1)
    alpha = getAlpha(beta)
    proposal = D.inverse_gamma.InverseGamma(alpha, beta)
    return -proposal.log_prob(noise_scale.square()) # (batch, n)


# def noiseIgExp(noise_param: torch.Tensor,
#                noise_std: torch.Tensor) -> torch.Tensor:
#     ''' Compute the MSE of the true noise std and its expected value under the noise parameters '''
#     # noise_param (batch, n)
#     # noise_std (batch, n)
#     alpha = getAlpha(noise_std)
#     beta = noise_param.squeeze(-1)
#     proposal = D.inverse_gamma.InverseGamma(alpha, beta)
#     expected_value = proposal.mean
#     return mse(noise_std.log(), expected_value.log()) # (batch, n)


def noiseLossWrapper(proposed: torch.Tensor,
                     noise_scale: torch.Tensor,
                     depths: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the noise loss function. ''' 
    # calculate losses for all dataset sizes and each beta
    b, n, _ = proposed.shape
    target = noise_scale.expand((b,n))
    losses = lf_noise(proposed, target).unsqueeze(-1)
    return averageOverN(losses, n, b, depths)


# -------- Kullback-Leibler Divergence
def klLossWrapper(mean_a: torch.Tensor, var_a: torch.Tensor,
                  mean_p: torch.Tensor, var_p: torch.Tensor,
                  beta: torch.Tensor,
                  depths: torch.Tensor):
    b, n, _ = mean_p.shape
    losses = torch.zeros(b, n, device=device)
    for i in range(b):
        mask = (beta[i] != 0.)
        mu_ai = mean_a[i, :, mask]
        mu_pi = mean_p[i, :, mask]
        Sigma_ai = torch.diag_embed(var_a[i, :, mask])
        Sigma_pi = torch.diag_embed(var_p[i, :, mask])
        post_a = D.multivariate_normal.MultivariateNormal(mu_ai, Sigma_ai)
        post_p = D.multivariate_normal.MultivariateNormal(mu_pi, Sigma_pi)
        losses[i] = D.kl.kl_divergence(post_a, post_p)
    return averageOverN(losses, n, b, depths)
 

# -------- histogram methods
grid = torch.linspace(-10, 10, steps=128)

def histMode(dist: torch.Tensor) -> torch.Tensor:
    # dist (b, n, 128, d)
    b, n, _, d = dist.shape
    grid_expanded = grid.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n, d, -1)
    index = dist.argmax(dim=-2).unsqueeze(-1) # (b, n, d, 1)
    return torch.gather(grid_expanded, dim=-1, index=index).squeeze(-1) # (b, n, d)


def histMean(dist: torch.Tensor) -> torch.Tensor:
    # dist (b, n, 128, d)
    return torch.matmul(dist.permute(0, 1, 3, 2), grid)


def histVariance(dist: torch.Tensor,
                 mean: torch.Tensor):
    # dist (b, n, 128, d)
    # mean (b, n, d)
    grid_expanded = grid.view(1, 1, -1, 1)
    mean_expanded = mean.unsqueeze(2)
    squared_diff = (grid_expanded - mean_expanded) ** 2
    weighted_squared_diff = squared_diff * dist
    return torch.sum(weighted_squared_diff, dim=2)
 

def histMSE(means: torch.Tensor,
            targets: torch.Tensor) -> torch.Tensor:
    # means (b, n, d)
    # targets (b, d)
    betas = targets.unsqueeze(1).expand_as(means)
    return mse(means, betas)


def histLogProb(dist: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
    # dist (b, n, 128, d)
    # targets (b, d)
    grid_expanded = grid.view(1, 1, -1)
    index = (grid_expanded - targets.unsqueeze(-1)).abs().argmin(dim=-1) # (b, d)
    index_expanded = index.unsqueeze(1).unsqueeze(1).expand(-1, dist.shape[1], -1, -1) # (b, n, 1, d)
    probs = torch.gather(dist, dim=2, index=index_expanded).squeeze(2) # (b, n, d)
    return -probs.log()


def histLossWrapper(probs: torch.Tensor,
                    targets: torch.Tensor,
                    depths: torch.Tensor) -> torch.Tensor:
    ''' Wrapper for the beta loss function.
    Handles the case 3D tensors (where the second dimension is the number of subjects = seq_size).
    Drop the losses for datasets that have fewer than n = noise_tol * number of features.'''
    # calculate losses for all dataset sizes and each beta
    b, n, _, _ = probs.shape
    # losses = lf_hist(probs, targets) # (b, n, d)
    means = histMean(probs)
    variances = histVariance(probs, means)
    losses = histMSE(means, targets) + 0.01 * variances
    return averageOverN(losses, n, b, depths)


# -------- training and testing methods
def run(models: tuple,
        batch: dict,
        num_examples: int = 0,
        unpad: bool = True,
        printer: Callable = print) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Run a batch through the model and return the loss. '''
    depths = batch["d"]
    y = batch["y"].to(device)
    X = batch["X"].to(device)
    beta = batch["beta"].float()
    inputs = torch.cat([y.unsqueeze(-1), X], dim=-1)
    mu_beta, stds_beta, stds_b = models[0](inputs)
    noise_inputs = torch.cat([y.unsqueeze(-1), mu_beta.detach(), stds_beta.detach()], dim=-1)
    noise_param = models[1](noise_inputs, True)

    # compute beta parameter loss per batch and predictor (optionally over multiple model outputs per batch)
    losses = betaLossWrapper(mu_beta, stds_beta, beta, depths) # (batch, n_predictors)
    if "rfx" not in batch:
        # compute mean loss over all batches and predictors (optionally ignoring padded predictors)
        loss = maskLoss(losses, beta) if unpad else losses.mean()
    else:
        # compute losses for random effects structure
        depths_rfx = batch["q"]
        rfx = batch["rfx"].to(device)
        stds_b_true = batch["S_emp"].to(device).sqrt()
        losses_rfx = rfxLossWrapper(stds_b, stds_b_true, rfx, depths_rfx) # (batch, n_predictors)
        # join losses
        losses_joint = torch.cat([losses, losses_rfx], dim=1)
        targets = torch.cat([beta, rfx[:,0]], dim=1)
        loss = maskLoss(losses_joint, targets) if unpad else losses.mean()
        
    # compute noise parameter loss per batch
    noise_std = batch["sigma_error"].unsqueeze(-1).float()
    losses_noise = noiseLossWrapper(noise_param, noise_std, depths) # (batch, 1)
    loss_noise = losses_noise.mean()

    # optionally print some examples
    for i in range(num_examples):
        mask = (beta[i] != 0.)
        beta_i = beta[i, mask].detach().numpy()        
        printer(f"\n{console_width * '-'}")
        printer(f"True       : {beta_i}")
        if cfg.type == "ffx":
            mu_i = batch["mu_n"][i, -1, mask].detach().numpy()
            printer(f"Analytical : {mu_i}")
        outputs_i = mu_beta[i, -1, mask].detach().numpy()
        printer(f"Predicted  : {outputs_i}")
        printer(f"{console_width * '-'}\n")
    return loss, loss_noise


def compare(models: nn.Module, batch: dict) -> torch.Tensor:
    ''' Compate analytical posterior with proposed posterior using KL divergence '''
    depths = batch["d"]
    y = batch["y"].to(device)
    X = batch["X"].to(device)
    beta = batch["beta"].float()
    inputs = torch.cat([y.unsqueeze(-1), X], dim=-1) # <eos> token?
    mean_proposed, std_proposed, _ = model(inputs)
    var_proposed = std_proposed.square()

    # get analytical posterior (posterior mean vector and covariance matrix)
    mean_analytical = batch["mu_n"].float().squeeze(-1)
    Sigma_analytical = batch["Sigma_n"].float()
    var_analytical = torch.diagonal(Sigma_analytical, dim1=-2, dim2=-1)
    
    # correct for noise variance
    sigma_error = batch["sigma_error"].float()
    var_analytical = sigma_error.square().unsqueeze(-1).unsqueeze(-1) * var_analytical
    
    # Compute KL divergences 
    losses = klLossWrapper(mean_analytical, var_analytical,
                           mean_proposed, var_proposed,
                           beta, depths)
    return losses.mean() # average over batch


def savePredictions(models: Tuple[nn.Module, nn.Module], batch: dict, iteration_index: int, batch_index: int) -> None:
    ''' save model outputs '''
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    inputs = torch.cat([y.unsqueeze(-1), X], dim=-1)
    mu, sigma, s = models[0](inputs)
    noise_inputs = torch.cat([y.unsqueeze(-1), mu.detach(), sigma.detach()], dim=-1)
    noise_params = models[1](noise_inputs, True)

    fname = Path(pred_path, f"predictions_i={iteration_index}_b={batch_index}.pt")
    out = {
        "means": mu,
        "stds": sigma,
        "s": s,
        "abs": noise_params,
    }
    torch.save(out, fname)


def train(models: Tuple[nn.Module, nn.Module],
          optimizers: Tuple[schedulefree.AdamWScheduleFree, schedulefree.AdamWScheduleFree],
          dataloader: DataLoader,
          writer: Union[SummaryWriter, None],
          logger: Union[Logger, None],
          iteration: int,
          step: int) -> int:
    ''' Train the model for a single iteration. '''
    for model in models:
        model.train()
    for optimizer in optimizers:
        optimizer.train()
    iterator = tqdm(dataloader, desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [T]")
    for batch in iterator:
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss, loss_noise = run(models, batch, unpad=True)
        loss.backward()
        loss_noise.backward()
        for optimizer in optimizers:
            optimizer.step()
        iterator.set_postfix({"loss": loss.item()})
        if writer is not None:
            writer.add_scalar("loss_train", loss.item(), step)
            writer.add_scalar("loss_train_noise", loss_noise.item(), step)
        if logger is not None:
            logger.write(iteration, step, loss.item(), "loss_train")
            logger.write(iteration, step, loss_noise.item(), "loss_train_noise")
        step += 1
    return step


def validate(models: Tuple[nn.Module, nn.Module],
             optimizers: Tuple[schedulefree.AdamWScheduleFree, schedulefree.AdamWScheduleFree],
             dataloader: DataLoader,
             writer: Union[SummaryWriter, None],
             logger: Union[Logger, None],
             iteration: int,
             step: int) -> int:
    ''' Validate the model for a single iteration. '''
    for model in models:
        model.eval()
    for optimizer in optimizers:
        optimizer.eval()
    iterator = tqdm(dataloader, desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [V]")
    with torch.no_grad():
        for j, batch in enumerate(iterator):
            # preset validation loss
            loss_val, loss_val_noise = run(models, batch, unpad=True, printer=iterator.write)
            iterator.set_postfix({"loss": loss_val.item()})
            if writer is not None:
                writer.add_scalar("loss_val", loss_val.item(), step)
                writer.add_scalar("loss_val_noise", loss_val_noise.item(), step)
            if logger is not None:
                logger.write(iteration, step, loss_val.item(), "loss_val")
                logger.write(iteration, step, loss_val_noise.item(), "loss_val_noise")

            # optionally calculate KL loss
            if cfg.kl and cfg.type == "ffx":
                loss_kl = compare(models[0], batch)
                if writer is not None:
                    writer.add_scalar("loss_kl", loss_kl.item(), step)
                if logger is not None:
                    logger.write(iteration, step, loss_kl.item(), "loss_kl")

            # optionally save predictions
            if iteration % 5 == 0 and not cfg.proto:
                savePredictions(models, batch, iteration, j)

            # optionally print predictions
            if iteration % 5 == 0 and j % 15 == 0:
                run(models, batch, unpad=True, printer=iterator.write, num_examples=1) # type: ignore
                
            step += 1
    return step


###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=0, help="Model seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use [cuda, cpu]")
    parser.add_argument("-p", "--preload", type=int, default=0, help="Preload model from iteration #p")
    parser.add_argument("--model-folder", type=str, default="checkpoints", help="Model folder")
    parser.add_argument("--proto", action="store_true", help="prototyping: don't log anything during (default = False)")

    # data
    parser.add_argument("-t", "--type", type=str, default="mfx", help="Type of dataset [ffx, mfx] (default = mfx)")
    parser.add_argument("-d", type=int, default=9, help="Number of predictors (with bias)")
    parser.add_argument("-m", type=int, default=50, help='Maximum number of samples to draw per linear model (default = 50).')
    parser.add_argument("-f", "--fixed", type=float, default=0., help='Fixed noise variance (default = 0. -> not fixed)')
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations to train (default = 10)")
    parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size (default = 50)")

    # model and loss
    parser.add_argument("-l", "--loss", type=str, default="logprob", help="Loss function [mse, logprob] (default = logprob)")
    parser.add_argument("--loss_rfx", type=str, default="mse", help="Loss function for rfx [mse, logprob] (default = logprob)")
    parser.add_argument("--loss_noise", type=str, default="logprob", help="Loss function for noise [mse, logprob, ig_exp] (default = logprob)")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (default = 0)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension (default = 128)")
    parser.add_argument("--ff", type=int, default=256, help="Feedforward dimension (transformer, default = 256)")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads (transformer, default = 8)")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers (transformer, default = 3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam, default = 1e-3)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon (Adam, default = 1e-8)")
    parser.add_argument("--kl", action="store_false", help="additionally report KL Divergence for validation set (default = True)")
    parser.add_argument("--tol", type=int, default=0, help="Noise tolerance: ignore all losses for n < noise_tol * d (default = 0)")

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
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # --- set up models
    model = TransformerDecoder(
                num_predictors=cfg.d,
                hidden_size=cfg.hidden,
                ff_size=cfg.ff,
                n_heads=cfg.heads,
                n_layers=cfg.layers,
                dropout=cfg.dropout,
                seed=cfg.seed).to(device)
    model_noise = TransformerDecoder(
                num_predictors= 2 * cfg.d,
                hidden_size=cfg.hidden,
                ff_size=cfg.ff,
                n_heads=cfg.heads,
                n_layers=1,
                dropout=cfg.dropout,
                seed=cfg.seed).to(device)
    models = (model, model_noise)
    print(f"Model: Transformer with {cfg.hidden} hidden units, " + \
            f"{cfg.ff} feedforward units, {cfg.heads} heads, {cfg.layers} layer(s), " + \
            f"{cfg.dropout} dropout")

    # --- set up optimizers
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr, eps=cfg.eps)
    optimizer_noise = schedulefree.AdamWScheduleFree(model_noise.parameters(), lr=cfg.lr, eps=cfg.eps)
    optimizers = (optimizer, optimizer_noise)

    # --- optionally preload a model
    initial_iteration, global_step, validation_step = 1, 1, 1
    if cfg.preload:
        initial_iteration, global_step, validation_step, timestamp = load(models, optimizers, cfg.preload)
        print(f"Preloaded model from iteration {cfg.preload}, starting at iteration {initial_iteration}.")

    # --- loss functions
    # 1. parameters
    if cfg.loss == "mse":
        lf = betaMSE
    elif cfg.loss == "logprob":
        lf = betaLogProb
    else:
        raise ValueError(f"Loss {cfg.loss} not recognized.")

    # 2. rfx
    if cfg.loss_rfx == "mse":
        lf_rfx = rfxMSE
    elif cfg.loss_rfx == "logprob":
        lf_rfx = rfxLogProb
    else:
        raise ValueError(f"Loss {cfg.loss_rfx} not recognized.")

    # 3. noise
    if cfg.loss_noise == "mse":
        lf_noise = noiseMSE
    elif cfg.loss_noise == "logprob":
        lf_noise = noiseLogProb
    elif cfg.loss_noise == "ig_exp":
        lf_noise = noiseIgExp
    else:
        raise ValueError(f"Loss {cfg.loss_noise} not recognized.")

    # --- logging
    if cfg.proto:
        writer, logger = None, None
    else:
        writer = SummaryWriter(Path("runs", modelID(cfg), timestamp))
        logger = Logger(Path("losses", modelID(cfg), timestamp))
    pred_path = Path("predictions", modelID(cfg), timestamp)
    pred_path.mkdir(parents=True, exist_ok=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}, Loss: {cfg.loss}, Learning rate: {cfg.lr}, Epsilon: {cfg.eps}, Seed: {cfg.seed}, Device: {device}")
    
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # training loop
    print("Preparing validation dataset...")
    fname = dsFilenameVal(cfg.type, cfg.d - 1, cfg.m, cfg.fixed)
    dataloader_val = getDataLoader(fname, batch_size=100)
    print(f"Training for {cfg.iterations + 1 - initial_iteration} iterations with 10k datasets per iteration and a batch size of {cfg.batch_size}...")
    for iteration in range(initial_iteration, cfg.iterations + 1):
        fname = dsFilename(cfg.type, cfg.d - 1, cfg.m, cfg.fixed, int(1e4), iteration)
        dataloader_train = getDataLoader(fname, cfg.batch_size)
        global_step = train(models, optimizers, dataloader_train, writer, logger, iteration, global_step)
        validation_step = validate(models, optimizers, dataloader_val, writer, logger, iteration, validation_step)
        save(models, optimizers, iteration, global_step, validation_step, timestamp)
 

