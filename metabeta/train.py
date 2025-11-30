import os
from pathlib import Path
import argparse
import yaml
import csv
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import schedulefree

from metabeta.utils import setDevice, dsFilename, getConsoleWidth
from metabeta.data.dataset import getDataLoader
from metabeta.models.approximators import Approximator, ApproximatorMFX


def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()
    
    # misc
    parser.add_argument('-s', '--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--device', type=str, default='mps', help='device to use [cpu, cuda, mps]')
    parser.add_argument('--cores', type=int, default=8, help='nubmer of processor cores to use (default = 8)')
    parser.add_argument('--plot', action='store_false', help='plot sampling results (default = True)')
    
    # loading
    parser.add_argument('--d_tag', type=str, default='all', help='suffix for data ID (default = '')')
    parser.add_argument('--m_tag', type=str, default='all', help='suffix for model ID (default = '')')
    parser.add_argument('--c_tag', type=str, default='config', help='name of model config file (default = "config")')
    parser.add_argument('-l', '--load', type=int, default=0, help='load model from iteration #p')
    
    # training
    parser.add_argument('-i', '--iterations', type=int, default=10, help='maximum number of iterations to train (default = 10)')
    parser.add_argument('--patience', type=int, default=20, help='early stopping criterium (default = 20)')
    parser.add_argument('--bs_train', type=int, default=4096, help='number of regression datasets per training set (default = 4096)')
    parser.add_argument('--bs_val', type=int, default=256, help='number of regression datasets in validation set (default = 256)')
    parser.add_argument('--bs_mini', type=int, default=16, help='umber of regression datasets per minibatch (default = 16)')
    parser.add_argument('--lr', type=float, default=5e-4, help='optimizer learning rate (default = 5e-4)')
    
    return parser.parse_args()


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

    def write(self, iteration: int, step: int, loss: float, losstype: str) -> None:
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


def totalNorm(model: nn.Module):
    total_norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2)
            for p in model.parameters()
            if p.grad is not None
        ]),
        p=2,
    )
    return total_norm

# -----------------------------------------------------------------------------
# loading and saving
def save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
    current_iteration: int,
    current_global_step: int,
    current_validation_step: int,
    timestamp: str,
) -> None:
    '''Save the model and optimizer state.'''
    fname = Path(model_path, f'checkpoint_i={current_iteration}.pt')
    torch.save(
        {
            'iteration': current_iteration,
            'global_step': current_global_step,
            'validation_step': current_validation_step,
            'timestamp': timestamp,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': model.stats,
        },
        fname,
    )


def load(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree,
    initial_iteration: int,
) -> tuple[int, int, int, str]:
    '''Load the model and optimizer state from a previous run,
    returning the initial iteration and seed.'''
    fname = Path(model_path, f'checkpoint_i={initial_iteration}.pt')
    print(f'Loading checkpoint from {fname}')
    state = torch.load(fname, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.stats = state['stats']
    initial_iteration = state['iteration'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
    validation_step = state['validation_step']
    timestamp = state['timestamp']
    return initial_iteration, global_step, validation_step, timestamp


# -----------------------------------------------------------------------------
# the bread and butter


def run(
    model: ApproximatorMFX,
    batch: dict[str, torch.Tensor],
    sample: bool = False,
) -> dict:
    targets, names, moments = {}, {}, {}
    results = model(batch, sample=sample, s=100)
    targets['global'] = model.targets(batch, local=False)
    targets['local'] = model.targets(batch, local=True)
    names['global'] = model.names(batch, local=False)
    names['local'] = model.names(batch, local=True)
    if sample:
        moments['global'] = model.moments(results['proposed']['global'])
        moments['local'] = model.moments(results['proposed']['local'])
    out = {
        'loss': results['loss'],
        'proposed': results['proposed'],
        'targets': targets,
        'names': names,
        'moments': moments,
    }
    return out


def train(
    model: Approximator,
    optimizer: schedulefree.AdamWScheduleFree,
    dl: DataLoader,
    step: int,
) -> int:
    iterator = tqdm(dl, desc=f'iteration {iteration:02d}/{cfg.iterations:02d} [T]')
    running_sum = 0.0
    model.train()
    optimizer.train()
    for i, batch in enumerate(iterator):
        optimizer.zero_grad(set_to_none=True)
        results = model(batch, sample=False)
        loss = results['loss'].mean()
        loss.backward()
        
        # safe gradient handling
        if torch.isfinite(totalNorm(model)):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            iterator.set_postfix_str('loss: skipped step due to NaNs')
            continue
        
        running_sum += loss.item()
        loss_train = running_sum / (i + 1)
        iterator.set_postfix_str(f'loss: {loss_train:.3f}')
        logger.write(iteration, step, loss_train, 'loss_train')
        step += 1
    return step


def validate(model: ApproximatorMFX, dl: DataLoader, step: int) -> int:
    results = None
    iterator = tqdm(dl, desc=f'iteration {iteration:02d}/{cfg.iterations:02d} [V]')
    loss_val = 0
    sample = False
    if step % 5 == 0:
        sample = True

    # run validation steps
    with torch.no_grad():
        model = model.to('cpu')
        model.eval()
        optimizer.eval()
        for i, batch in enumerate(iterator):
            results = run(model, batch, sample=sample)
            loss = results['loss'].mean().item()  # type: ignore
            loss_val += loss if isinstance(loss, float) else 0
            iterator.set_postfix_str(f'loss: {loss_val / (i + 1):.3f}')
        loss_val /= len(iterator)
        step += 1
        logger.write(iteration, step, loss_val, 'loss_val')
        stopper.update(loss_val)

        # evaluate samples
        if sample:
            assert results is not None
            # global results
            rmse, r = plot.recovery(  # type: ignore
                targets=results['targets']['global'],
                names=results['names']['global'],
                means=results['moments']['global'][0],
                return_stats=True,
            )
            logger.write(iteration, step, rmse, 'rmse')
            logger.write(iteration, step, r, 'r')
            iterator.write(f'Global - RMSE: {rmse:.3f}, R: {r:.3f}')

            # local results
            if 'local' in results['names']:
                rmse, r = plot.recovery(  # type: ignore
                    targets=results['targets']['local'],
                    names=results['names']['local'],
                    means=results['moments']['local'][0],
                    return_stats=True,
                )
                iterator.write(f'Local - RMSE: {rmse:.3f}, R: {r:.3f}')

        model = model.to(device)
    return step


###############################################################################
if __name__ == '__main__':
    # --- setup training config
    cfg = setup()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    console_width = getConsoleWidth()
    device = setDevice(cfg.device)
    torch.set_num_threads(cfg.cores)
    
    # --- setup model
    with open(Path('models', f'{cfg.c_tag}.yaml'), 'r') as f:
        model_cfg = yaml.safe_load(f)
        model_cfg['general']['seed'] = cfg.seed
        model_cfg['general']['tag'] = cfg.m_tag
    model = ApproximatorMFX.build(model_cfg).to(device)
    print(f'{'-' * console_width}\nmodel: {model.id}\nLearning rate: {cfg.lr}\nDevice: {device}')

    # --- set up optimizer
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr)

    # --- optionally preload a model
    model_path = Path('outputs', 'checkpoints', model.id)
    model_path.mkdir(parents=True, exist_ok=True)
    initial_iteration, global_step, validation_step = 1, 1, 1
    if cfg.preload:
        initial_iteration, global_step, validation_step, timestamp = load(
            model, optimizer, cfg.preload
        )
        print(f'preloaded model from iteration {cfg.preload}, starting at iteration {initial_iteration}...')

    # --- logging and stopping
    log_path = Path('outputs', 'losses', model.id, timestamp)
    logger = Logger(log_path)
    stopper = EarlyStopping(patience=cfg.patience)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {num_params}, summarizer: {model.num_sum}, posterior: {model.num_inf}')

    # -------------------------------------------------------------------------
    # training loop
    print(f'fixed effects: {cfg.d}\nrandom effects: {cfg.q}\nobservations (max): {cfg.n}')
    fn = dsFilename(
        cfg.fx_type,
        'val',
        1, cfg.m, cfg.n, cfg.d, cfg.q, cfg.bs_val,
        # varied=cfg.varied,
        tag=cfg.d_tag,
    )
    dl_val = getDataLoader(
        fn,
        cfg.bs_val,
        max_d=cfg.d,
        max_q=cfg.q,
        permute=False,
        autopad=True,
        device='cpu',
    )

    if cfg.preload > 0:
        iteration = cfg.preload
        validate(model, dl_val, cfg.preload)

    print(f'iterations: {cfg.iterations + 1 - initial_iteration}\npatience: {cfg.patience}\nbatches per iteration: 200\ndatasets per batch: {cfg.bs_mini}\n{'-' * console_width}')
    for iteration in range(initial_iteration, cfg.iterations + 1):
        fn = dsFilename(
            cfg.fx_type,
            'train',
            cfg.bs_mini, cfg.m, cfg.n, cfg.d, cfg.q, cfg.bs_train,
            part=iteration,
            # varied=cfg.varied,
            tag=cfg.d_tag,
        )
        dl_train = getDataLoader(
            fn,
            cfg.bs_mini // 2,
            max_d=cfg.d,
            max_q=cfg.q,
            permute=cfg.permute,
            autopad=False,
            device=device,
        )
        global_step = train(model, optimizer, dl_train, global_step)
        validation_step = validate(model, dl_val, validation_step)
        if iteration % 5 == 0 or stopper.stop:
            save(model, optimizer, iteration, global_step, validation_step, timestamp)
        if stopper.stop:
            break


