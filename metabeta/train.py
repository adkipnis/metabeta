import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import schedulefree

from metabeta.models.approximator import (
    Approximator, SummarizerConfig, PosteriorConfig, ApproximatorConfig)
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import setDevice, datasetFilename


def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()
 
    # misc
    parser.add_argument('-s', '--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--reproducible', action='store_false', help='use deterministic learning trajectory (default = True)')
    parser.add_argument('--cores', type=int, default=8, help='number of processor cores to use (default = 8)')
    parser.add_argument('--device', type=str, default='cpu', help='device to use [cpu, cuda, mps], (default = mps)')
    # parser.add_argument('--save', type=int, default=10, help='save model every #p iterations (default = 10)')

    # model
    parser.add_argument('--ds_type', type=str, default='toy', help='type of predictors [toy, flat, scm, sampled], (default = toy)')
    parser.add_argument('--cfg', type=str, default='default', help='name of model config yml file')
    parser.add_argument('-l', '--load', type=int, default=0, help='load model from iteration #l')
    parser.add_argument('--compile', action='store_true', help='compile model (default = False)')
 
    # data dimensions
    parser.add_argument('-d', '--max_d', type=int, default=3, help='maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 16).')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).')
    parser.add_argument('--min_m', type=int, default=5, help='minimum number of groups (default = 5).')
    parser.add_argument('--max_m', type=int, default=30, help='maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='minimum number of samples per group (default = 10).')
    parser.add_argument('--max_n', type=int, default=70, help='maximum number of samples per group (default = 70).')

    # training & testing
    parser.add_argument('-e', '--max_epochs', type=int, default=10, help='maximum number of epochs to train (default = 10)')
    parser.add_argument('--bs_mini', type=int, default=32, help='number of regression datasets per training minibatch (default = 32)')
    parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate (default = 1e-3)')
    parser.add_argument('--test_interval', type=int, default=5, help='sample posterior every #n epochs (default = 5)')
    # parser.add_argument('--patience', type=int, default=20, help='early stopping criterium (default = 20)')

    return parser.parse_args()

# -----------------------------------------------------------------------------
def setSeed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def makeReproducible() -> None:
    torch.use_deterministic_algorithms(True)
    if ARGS.device == 'mps':
        ARGS.device = 'cpu'
    elif ARGS.device == 'cuda':
        torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------
def getRmse(proposed: dict[str, dict[str, torch.Tensor]],
            data: dict[str, torch.Tensor],
            ) -> dict[str, float]:
    d = ARGS.max_d

    # targets
    rfx = data['rfx']
    ffx = data['ffx']
    sigma_rfx = data['sigma_rfx']
    sigma_eps = data['sigma_eps']
 
    # means
    means = proposed['global']['samples'].mean(-2)
    ffx_mean = means[..., :d]
    sigma_rfx_mean = means[..., d:-1]
    sigma_eps_mean = means[..., -1]
    rfx_mean = proposed['local']['samples'].mean(-2)

    # rmses
    rmses = {}
    se_ffx = F.mse_loss(ffx, ffx_mean, reduction='none')
    rmses['ffx'] = torch.sqrt(se_ffx.sum() / data['mask_d'].sum()).item()
    se_sr = F.mse_loss(sigma_rfx, sigma_rfx_mean, reduction='none')
    rmses['sigma_rfx'] = torch.sqrt(se_sr.sum() / data['mask_q'].sum()).item()
    rmses['sigma_eps'] = torch.sqrt(F.mse_loss(sigma_eps, sigma_eps_mean)).item()
    mask_rfx = data['mask_m'].unsqueeze(-1) * data['mask_q'].unsqueeze(-2)
    se_rfx = F.mse_loss(rfx, rfx_mean, reduction='none')
    rmses['rfx'] = torch.sqrt(se_rfx.sum() / mask_rfx.sum()).item()

    return rmses

# -----------------------------------------------------------------------------
def train(
    model: Approximator,
    optimizer: schedulefree.AdamWScheduleFree,
    dl: Dataloader,
    epoch: int,
) -> None:
    iterator = tqdm(dl, desc=f'Epoch {epoch:02d}/{ARGS.max_epochs:02d} [T]')
    running_sum = 0.0
    model.train()
    optimizer.train()
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(iterator):
        # get loss
        batch = toDevice(batch, model.device)
        loss = model.forward(batch)
        loss = loss['total'].mean()

        # calculate accumulated gradient with clipped norm
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # log loss
        running_sum += loss.item()
        loss_train = running_sum / (i + 1)
        iterator.set_postfix_str(f'NLL: {loss_train:.3f}')

@torch.no_grad()
def valid(
    model: Approximator,
    optimizer: schedulefree.AdamWScheduleFree,
    dl: Dataloader,
    epoch: int = 0,
) -> None:
    iterator = tqdm(dl, desc=f'Epoch {epoch:02d}/{ARGS.max_epochs:02d} [V]')
    running_sum = 0.0
    model.eval()
    optimizer.eval()
    for i, batch in enumerate(iterator):
        batch = toDevice(batch, model.device)
        loss = model.forward(batch)
        loss = loss['total'].mean()
        running_sum += loss.item()
        loss_valid = running_sum / (i + 1)
        iterator.set_postfix_str(f'NLL: {loss_valid:.3f}')

@torch.no_grad()
def test(
    model: Approximator,
    optimizer: schedulefree.AdamWScheduleFree,
    dl: Dataloader,
    epoch: int = 0,
) -> None:
    # expects single batch from dl
    iterator = tqdm(dl, desc=f'Epoch {epoch:02d}/{ARGS.max_epochs:02d} [S]')
    model.eval()
    optimizer.eval()
    for batch in iterator:
        batch = toDevice(batch, model.device)
        proposed = model.estimate(batch, n_samples=200)
        rmses = getRmse(proposed, batch)
        rmse_str = ', '.join([f'{k}={v:.3f}' for k,v in rmses.items()])
        iterator.set_postfix_str(f'RMSE: {rmse_str}')


# =============================================================================
if __name__ == '__main__':
    # --- setup training config
    ARGS = setup()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    torch.set_num_threads(ARGS.cores)

    # reproducibility
    if ARGS.reproducible:
        makeReproducible()
    setSeed(ARGS.seed)
    device = setDevice(ARGS.device)

    # model
    s_cfg = SummarizerConfig(
        d_model=64,
        d_ff=128,
        d_output=32,
        n_blocks=2,
        n_isab=0,
    )
    p_cfg = PosteriorConfig(
        transform='spline',
        subnet_kwargs={
            'net_type': 'mlp',
            'd_ff': 128,
            'depth': 3,
            'activation': 'ELU',
            'zero_init': True,
        },
        n_blocks=3,
    )
    cfg = ApproximatorConfig(
        d_ffx=3,
        d_rfx=1,
        summarizer=s_cfg,
        posterior=p_cfg,
    )
    model = Approximator(cfg).to(device)
    if ARGS.compile and device.type != 'mps':
        model = model.compile()

    # optimizer
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=ARGS.lr)
 
    # validation and test data
    path_valid = Path('outputs', 'data', datasetFilename(ARGS, 'valid'))
    dl_valid = Dataloader(path_valid)
    path_test = Path('outputs', 'data', datasetFilename(ARGS, 'test'))
    dl_test = Dataloader(path_test)

    # reference run
    print('Performance before training:')
    valid(model, optimizer, dl_valid)
    test(model, optimizer, dl_test)

    # loop
    print(f'\nTraining for {ARGS.max_epochs} epochs...')
    for epoch in range(1, ARGS.max_epochs + 1):
        path_train = Path('outputs', 'data', datasetFilename(ARGS, 'train', epoch))
        dl_train = Dataloader(path_train, batch_size=ARGS.bs_mini)
        train(model, optimizer, dl_train, epoch)
        valid(model, optimizer, dl_valid, epoch)
        if epoch % ARGS.test_interval == 0:
            test(model, optimizer, dl_test, epoch)
