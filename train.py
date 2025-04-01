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
from utils import dsFilename, dsFilenameVal
from dataset import LMDataset
from models import TransformerEncoder, TransformerDecoder
from proposal import DiscreteProposal, MixtureProposal, normalBins, halfNormalBins
mse = nn.MSELoss(reduction='none')

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


def getDataLoader(filename: Path, batch_size: int) -> DataLoader:
    ''' Load a dataset from a file,
    split into train and validation set and return a DataLoader. '''
    assert filename.exists(), f"File {filename} does not exist, you must generate it first using generate.py"
    ds_raw = torch.load(filename, weights_only=False)
    ds = LMDataset(**ds_raw, permute=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def modelID(cfg: argparse.Namespace) -> str:
    ''' Return a string that identifies the model. '''
    noise = "variable" if cfg.fixed == 0 else cfg.fixed
    return f"{cfg.posterior_type}-{cfg.c}-transformer-{cfg.hidden}-{cfg.ff}-{cfg.heads}-{cfg.layers}-dropout={cfg.dropout}-loss={cfg.loss_ffx}-seed={cfg.seed}-fx={cfg.fx_type}-noise={noise}"
    

def getCheckpointPath(iteration: int) -> Path:
    ''' Get the filename for the model checkpoints. '''
    model_id = modelID(cfg)
    model_filename = f"{model_id}-{iteration:02d}.pt"
    return Path("outputs", cfg.model_folder, model_filename)


def save(models: Tuple[nn.Module, nn.Module],
         optimizers: Tuple[schedulefree.AdamWScheduleFree,
                           schedulefree.AdamWScheduleFree],
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
        'model_state_dict': models[0].state_dict(),
        'model_state_dict_noise': models[1].state_dict(),
        'optimizer_state_dict': optimizers[0].state_dict(),
        'optimizer_state_dict_noise': optimizers[1].state_dict(),
    }, model_filename)


def load(models: Tuple[nn.Module, nn.Module],
         optimizers: Tuple[schedulefree.AdamWScheduleFree,
                           schedulefree.AdamWScheduleFree],
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
# def chooseLossFn(target_type: str, posterior_type: str):
#     loss_type = eval(f"cfg.loss_{target_type}")
#     if loss_type == "mse":
#         return eval(f"{posterior_type}MSE")
#     elif loss_type == "logprob":
#         return eval(f"{posterior_type}LogProb")
#     else:
#         raise ValueError(f'loss type: "{loss_type}" not found.')
#

def averageOverN(losses: torch.Tensor,
                 n: int,
                 b: int,
                 depths: torch.Tensor,
                 n_min: int = 0,
                 weigh: bool = False) -> torch.Tensor:
    ''' mask out first n_min losses per batch,
    then calculate weighted average over n with higher emphasis on later n'''
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
    ''' Ignore padding values before aggregating the losses
        over batches and dims'''
    # losses (batch, d)
    mask = (targets != pad_val).float()
    masked_losses = losses * mask
    loss = masked_losses.sum() / mask.sum()
    return loss # (,)


# -------- methods for posteriors
def discreteLossWrapper(outputs: Dict[str, torch.Tensor],
                        targets: torch.Tensor,
                        depths: torch.Tensor,
                        target_type: str) -> torch.Tensor:
    # calculate losses for all dataset sizes and each beta
    logits = outputs[f"{target_type}_logits"]
    b, n, d, _ = logits.shape
    targets_exp = targets.unsqueeze(1).expand(b, n, d)
    loss_type = eval(f"cfg.loss_{target_type}")
    prop = dprop_normal if target_type == "ffx" else dprop_half_normal
    losses = prop.loss(loss_type, targets_exp, logits)
    return averageOverN(losses, n, b, depths)


def mixLossWrapper(outputs: Dict[str, torch.Tensor], 
                   target: torch.Tensor,
                   depths: torch.Tensor,
                   target_type: str) -> torch.Tensor:
    # calculate losses for all dataset sizes and each beta
    locs = outputs[f"{target_type}_loc"]
    scales = outputs[f"{target_type}_scale"]
    weights = outputs[f"{target_type}_weight"]
    b, n, _, _ = locs.shape
    if target_type == "rfx":
        mask = (target == 0.).float()
        target = (target + mask).log()
    elif target_type == "noise":
        target = target.log()
    loss_type = eval(f"cfg.loss_{target_type}")
    losses = mprop.loss(loss_type, locs, scales, weights, target, target_type)
    return averageOverN(losses, n, b, depths)


def discreteExamples(num_examples: int,
                     ffx: torch.Tensor,
                     outputs: Dict[str, torch.Tensor],
                     printer: Callable) -> None:
    for i in range(num_examples):
        mask = (ffx[i] != 0.)
        beta_i = ffx[i, mask].detach().numpy()        
        logits_masked = outputs["ffx_logits"][..., mask, :]
        mean = dprop_normal.mean(logits_masked)
        var = dprop_normal.variance(logits_masked, mean)
        mean_i = mean[i, -1].detach().numpy()
        sd_i = var[i, -1].sqrt().detach().numpy()
        printer(f"\n{console_width * '-'}")
        printer(f"True : {beta_i}")
        printer(f"Mean : {mean_i}")
        printer(f"SD   : {sd_i}")
        printer(f"{console_width * '-'}\n")


def mixExamples(num_examples: int,
                ffx: torch.Tensor,
                outputs: Dict[str, torch.Tensor],
                printer: Callable) -> None:
    if num_examples == 0:
        return
    locs, scales, weights = outputs["ffx_loc"], outputs["ffx_scale"], outputs["ffx_weight"]
    loc = mprop.mean(locs, scales, weights)
    scale = mprop.variance(locs, scales, weights, loc).sqrt()
    for i in range(num_examples):
         mask = (ffx[i] != 0.)
         beta_i = ffx[i, mask].detach().numpy()
         mean_i = loc[i, -1, mask].detach().numpy()
         std_i = scale[i, -1, mask].detach().numpy()
         printer(f"\n{console_width * '-'}")
         printer(f"True  : {beta_i}")
         printer(f"Mean  : {mean_i}")
         printer(f"SD    : {std_i}")
         printer(f"{console_width * '-'}\n")


# -------- training and testing methods
def assembleNoiseInputs(y: torch.Tensor, X: torch.Tensor,
                        outputs: Dict[str, torch.Tensor],
                        posterior_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if posterior_type == "discrete":
        logits = outputs["ffx_logits"]
        loc = dprop_normal.mean(logits).detach()
        scale = dprop_normal.variance(logits, loc).detach().sqrt()
    elif posterior_type == "mixture":
        locs = outputs["ffx_loc"]
        scales = outputs["ffx_scale"]
        weights = outputs["ffx_weight"]
        loc = mprop.mean(locs, scales, weights).detach()
        scale = mprop.variance(locs, scales, weights, loc).sqrt().detach()
    else:
        raise ValueError(f"posterior type {posterior_type} not supported.")
    y_pred = torch.sum(X * loc, dim=-1).unsqueeze(-1)
    res = y - y_pred
    return res, scale


def locScaleWeights(outputs: torch.Tensor,
                    prefix: str) -> Dict[str, torch.Tensor]:
    loc = outputs[..., 0]
    scale = outputs[..., 1].exp()
    weights = nn.functional.softmax(outputs[..., 2], dim=-1)
    return {f"{prefix}_loc": loc,
            f"{prefix}_scale": scale,
            f"{prefix}_weight": weights}


def parseOutputs(outputs: torch.Tensor,
                 posterior_type: str,
                 c: int,
                 target_type: str) -> Dict[str, torch.Tensor]:
    b, n, d, _ = outputs.shape
    outputs = outputs.reshape(b, n, d, c, -1)
    if posterior_type == "discrete":
        ffx_dict = {f"{target_type}_logits": outputs[..., 0]}
        if outputs.shape[-1] == 1:
            return ffx_dict
        rfx_dict = {"rfx_logits": outputs[..., 1]}
    elif posterior_type == "mixture":
        ffx_dict = locScaleWeights(outputs[..., :3], target_type)
        if outputs.shape[-1] == 3:
            return ffx_dict
        rfx_dict = locScaleWeights(outputs[..., 3:], "rfx")
    else:
        raise ValueError(f'posterior type "{posterior_type}" not supported.')
    return {**ffx_dict, **rfx_dict}


def run(models: tuple,
        batch: dict,
        posterior_type: str,
        num_examples: int = 0,
        printer: Callable = print,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Run a batch through the model and return the loss. '''
    lossWrapper, examples = mixLossWrapper, mixExamples
    if posterior_type == "discrete":
        lossWrapper, examples = discreteLossWrapper, discreteExamples

    # unpack data
    ffx_depths = batch["d"]
    ffx = batch["ffx"].to(device)
    y = batch["y"].to(device)
    X = batch["X"].to(device)
    Z = batch["Z"].to(device)
    groups = batch["groups"].to(device)

    # estimate parameters and compute ffx loss
    outputs = models[0](y, X, Z, groups)
    output_dict = parseOutputs(outputs, posterior_type, cfg.c, "ffx")
    losses_ffx = lossWrapper(output_dict, ffx, ffx_depths, "ffx")
    

    # optionally compute rfx loss
    if cfg.fx_type == "mfx":
        rfx_depths = batch["q"]
        rfx = batch["rfx"].to(device)
        rfx_scale = batch["S"].to(device).sqrt()
        losses_rfx = lossWrapper(output_dict, rfx_scale, rfx_depths, "rfx") 
        losses_joint = torch.cat([losses_ffx, losses_rfx], dim=1)
        target_blueprint = torch.cat([ffx, rfx[:,0]], dim=1)
        loss = maskLoss(losses_joint, target_blueprint)
    else:
        loss = maskLoss(losses_ffx, ffx)

    # compute noise parameter loss per batch
    res, scale = assembleNoiseInputs(y, X, output_dict, posterior_type)
    noise_outputs = models[1](res, scale, Z, groups)
    noise_output_dict = parseOutputs(noise_outputs, posterior_type, cfg.c, "noise")

    # compute noise loss
    noise_std = batch["sigma_error"].to(device).unsqueeze(-1).float()
    losses_noise = lossWrapper(noise_output_dict, noise_std, ffx_depths, "noise")
    loss_noise = losses_noise.mean()

    # optionally print some examples
    examples(num_examples, ffx, output_dict, printer)
    return loss, loss_noise


def savePredictions(models: Tuple[nn.Module, nn.Module],
                    batch: dict,
                    posterior_type: str, 
                    iteration_index: int,
                    batch_index: int) -> None:
    y = batch["y"].to(device)
    X = batch["X"].to(device)
    Z = batch["Z"].to(device)
    groups = batch["groups"].to(device)
    outputs = models[0](y, X, Z, groups)
    output_dict = parseOutputs(outputs, posterior_type, cfg.c, "ffx")
    res, scale = assembleNoiseInputs(y, X, output_dict, posterior_type)
    noise_outputs = models[1](res, scale, Z, groups)
    noise_output_dict = parseOutputs(noise_outputs, posterior_type, cfg.c, "noise")
    fname = Path(pred_path, f"predictions_i={iteration_index}_b={batch_index}.pt")
    out = {**output_dict, **noise_output_dict}
    torch.save(out, fname)


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
        post_a = D.MultivariateNormal(mu_ai, Sigma_ai)
        post_p = D.MultivariateNormal(mu_pi, Sigma_pi)
        losses[i] = D.kl.kl_divergence(post_a, post_p)
    return averageOverN(losses, n, b, depths)


def compare(models: tuple, batch: dict) -> torch.Tensor:
    ''' Compate analytical posterior with proposed posterior using KL divergence '''
    depths = batch["d"]
    y = batch["y"].to(device)
    X = batch["X"].to(device)
    Z = batch["Z"].to(device)
    groups = batch["groups"].to(device)
    beta = batch["ffx"].float()
    outputs = models[0](y, X, Z, groups)
    output_dict = parseOutputs(outputs, "mixture", 1, "ffx")
    mean_proposed, var_proposed = output_dict["ffx_loc"].squeeze(-1), output_dict["ffx_scale"].squeeze(-1).square()

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


# -------- outer loops
def train(models: Tuple[nn.Module, nn.Module],
          optimizers: Tuple[schedulefree.AdamWScheduleFree,
                            schedulefree.AdamWScheduleFree],
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
    iterator = tqdm(dataloader,
                    desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [T]")
    for batch in iterator:
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss, loss_noise = run(models, batch, cfg.posterior_type,
                               printer=iterator.write)
        loss.backward()
        loss_noise.backward()
        # for model in models:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
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
             optimizers: Tuple[schedulefree.AdamWScheduleFree,
                               schedulefree.AdamWScheduleFree],
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
    iterator = tqdm(dataloader,
                    desc=f"iteration {iteration:02d}/{cfg.iterations:02d} [V]")
    with torch.no_grad():
        for j, batch in enumerate(iterator):
            # preset validation loss
            loss, loss_noise = run(models, batch, cfg.posterior_type,
                                   printer=iterator.write)
            iterator.set_postfix({"loss": loss.item()})
            if writer is not None:
                writer.add_scalar("loss_val", loss.item(), step)
                writer.add_scalar("loss_val_noise", loss_noise.item(), step)
            if logger is not None:
                logger.write(iteration, step, loss.item(), "loss_val")
                logger.write(iteration, step, loss_noise.item(), "loss_val_noise")

            # optionally calculate KL loss
            if cfg.kl and cfg.fx_type == "ffx" and cfg.c == 1 and cfg.posterior_type == "mixture":
                loss_kl = compare(models, batch)
                if writer is not None:
                    writer.add_scalar("loss_kl", loss_kl.item(), step)
                if logger is not None:
                    logger.write(iteration, step, loss_kl.item(), "loss_kl")

            # optionally save predictions
            if iteration % 5 == 0 and not cfg.proto:
                savePredictions(models, batch, cfg.posterior_type, iteration, j)

            # optionally print predictions
            if iteration % 5 == 0 and j % 30 == 0:
                run(models, batch, cfg.posterior_type,
                    printer=iterator.write, num_examples=1)

            step += 1
    return step


###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=0,help="Model seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use [cuda, cpu]")
    parser.add_argument("-p", "--preload", type=int, default=0, help="Preload model from iteration #p")
    parser.add_argument("--model-folder", type=str, default="checkpoints", help="Model folder")
    parser.add_argument("--proto", action="store_true", help="prototyping: don't log anything during (default = False)")

    # data
    parser.add_argument("-t", "--fx_type", type=str, default="mfx", help="Type of dataset [ffx, mfx] (default = ffx)")
    parser.add_argument("-d", type=int, default=8, help="Number of predictors (without bias, default = 8)")
    parser.add_argument("-n", type=int, default=50, help="Maximum number of samples to draw per linear model (default = 50).")
    parser.add_argument("-f", "--fixed", type=float, default=0., help="Fixed noise variance (default = 0. -> not fixed)")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations to train (default = 10)")
    parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size (default = 50)")

    # model and loss
    parser.add_argument("--posterior_type", type=str, default="mixture", help="Posterior architecture [discrete, mixture] (default = mixture)")
    parser.add_argument("-c", type=int, default=5, help="Number of mixture components resp. grid bins (default = 5)")
    parser.add_argument("--loss_ffx", type=str, default="nll", help="Loss function [mse, nll] (default = nll)")
    parser.add_argument("--loss_rfx", type=str, default="nll", help="Loss function for rfx [mse, nll] (default = nll)")
    parser.add_argument("--loss_noise", type=str, default="nll", help="Loss function for noise [mse, nll] (default = nll)")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (default = 0)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension (default = 256)")
    parser.add_argument("--ff", type=int, default=256, help="Feedforward dimension (transformer, default = 512)")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads (transformer, default = 4)")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers (transformer, default = 6)")
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
    device = torch.device("cuda"
                          if torch.cuda.is_available() and cfg.device == "cuda"
                          else "cpu")

    # --- set up models
    model = TransformerEncoder(
                n_inputs=1+2*(1+cfg.d) if cfg.fx_type == "mfx" else 2+cfg.d, # y, X, Z
                n_predictors=1+cfg.d,
                hidden_size=cfg.hidden,
                ff_size=cfg.ff,
                n_heads=cfg.heads,
                n_layers=cfg.layers,
                dropout=cfg.dropout,
                seed=cfg.seed,
                fx_type=cfg.fx_type,
                posterior_type=cfg.posterior_type,
                n_components=cfg.c).to(device)
    model_noise = TransformerEncoder(
                n_inputs=2+cfg.d, # residuals, scale
                n_predictors=1+cfg.d,
                hidden_size=cfg.hidden,
                ff_size=cfg.ff,
                n_heads=cfg.heads,
                n_layers=cfg.layers,
                dropout=cfg.dropout,
                seed=cfg.seed,
                posterior_type=f"{cfg.posterior_type}_noise",
                n_components=cfg.c).to(device)
    models = (model, model_noise)
    print(f"Model: {modelID(cfg)}")

    # --- set up proposal distributions
    if cfg.posterior_type == "discrete":
        normal_bins = normalBins(3., cfg.c+1)
        half_normal_bins = halfNormalBins(3., cfg.c+1)
        dprop_normal = DiscreteProposal(normal_bins)
        dprop_half_normal = DiscreteProposal(half_normal_bins)
    else:
        mprop = MixtureProposal()

    # --- set up optimizers
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(),
                                               lr=cfg.lr, eps=cfg.eps)
    optimizer_noise = schedulefree.AdamWScheduleFree(model_noise.parameters(),
                                                     lr=cfg.lr, eps=cfg.eps)
    optimizers = (optimizer, optimizer_noise)

    # --- optionally preload a model
    initial_iteration, global_step, validation_step = 1, 1, 1
    if cfg.preload:
        initial_iteration, global_step, validation_step, timestamp = \
            load(models, optimizers, cfg.preload)
        print(f"Preloaded model from iteration {cfg.preload}, starting at iteration {initial_iteration}.")

    # --- logging
    if cfg.proto:
        writer, logger = None, None
    else:
        writer = SummaryWriter(Path("outputs", "runs", modelID(cfg), timestamp))
        logger = Logger(Path("outputs", "losses", modelID(cfg), timestamp))
    pred_path = Path("outputs", "predictions", modelID(cfg), timestamp)
    pred_path.mkdir(parents=True, exist_ok=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_noise = sum(p.numel() for p in model_noise.parameters() if p.requires_grad)
    print(f"Total number of parameters: {num_params + num_params_noise}, FFX Loss: {cfg.loss_ffx}, RFX Loss: {cfg.loss_rfx}, Noise Loss: {cfg.loss_noise}, Learning rate: {cfg.lr}, Epsilon: {cfg.eps}, Device: {device}")
    
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # training loop
    print("Preparing validation dataset...")
    fname = dsFilenameVal(cfg.fx_type, cfg.d, cfg.n, cfg.fixed)
    dataloader_val = getDataLoader(fname, batch_size=100)
    print(f"Training for {cfg.iterations + 1 - initial_iteration} iterations with 10k datasets per iteration and a batch size of {cfg.batch_size}...")
    for iteration in range(initial_iteration, cfg.iterations + 1):
        fname = dsFilename(cfg.fx_type, cfg.d, cfg.n, cfg.fixed, int(1e4), iteration)
        dataloader_train = getDataLoader(fname, cfg.batch_size)
        global_step = train(models, optimizers, dataloader_train, writer,
                            logger, iteration, global_step)
        validation_step = validate(models, optimizers, dataloader_val, writer,
                                   logger, iteration, validation_step)
        if iteration % 5 == 0:
            save(models, optimizers, iteration, global_step, validation_step, timestamp)
 
