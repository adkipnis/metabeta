import yaml
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import schedulefree

from metabeta.utils.io import setDevice, datasetFilename, checkpointFilename
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.models.approximator import Approximator

logger = logging.getLogger(__name__)


def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()
 
    # misc
    parser.add_argument('--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--device', type=str, default='cpu', help='device to use [cpu, cuda, mps], (default = cpu)')
    parser.add_argument('--cores', type=int, default=8, help='number of processor cores to use (default = 8)')
    parser.add_argument('--reproducible', action='store_false', help='use deterministic learning trajectory (default = False)')
    parser.add_argument('--tb', action='store_false', help='enable tensorboard logging (default = False)')

    # model & optimizer
    parser.add_argument('--m_tag', type=str, default='toy', help='name of model config file')
    parser.add_argument('--n_samples', type=int, default=200, help='number of samples to draw from posterior on test set')
    parser.add_argument('--compile', action='store_true', help='compile model (default = False)')
    parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate (default = 1e-3)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip grad norm to this value (default = 1.0)')

    # data
    parser.add_argument('--d_tag', type=str, default='toy', help='name of data config file')
    parser.add_argument('--bs', type=int, default=32, help='number of regression datasets per training minibatch (default = 32)')

    # steps
    parser.add_argument('--skip_ref', action='store_true', help='skip the reference run before training (default = False)')
    parser.add_argument('-e', '--max_epochs', type=int, default=10, help='maximum number of epochs to train (default = 10)')
    # parser.add_argument('--patience', type=int, default=20, help='early stopping criterium (default = 20)')
    parser.add_argument('--test_interval', type=int, default=5, help='sample posterior every #n epochs (default = 5)')

    # saving & loading
    parser.add_argument('--save_latest', action='store_false', help='save latest model after each epoch (default = False)')
    parser.add_argument('--save_best', action='store_false', help='track and save best model wrt. validation set (default = False)')
    parser.add_argument('--load_latest', action='store_false', help='load latest model before training (default = False)')
    parser.add_argument('--load_best', action='store_false', help='load best model wrt. validation set (overwrites load_latest, default = False)')

    return parser.parse_args()

# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg

        # reproducibility
        if cfg.reproducible:
            self._reproducible()
        setSeed(cfg.seed)

        # misc setup
        self.device = setDevice(self.cfg.device)
        torch.set_num_threads(cfg.cores)

        # init data, model and optimizer
        self._initData()
        self._initModel()

        # checkpoint dir
        self.ckpt_dir = Path('..', 'outputs', 'checkpoints', self.cfg.m_tag)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        # tracking
        self.best_valid = float('inf')
        self.best_epoch = 0

    def _reproducible(self) -> None:
        torch.use_deterministic_algorithms(True)
        if self.cfg.device == 'mps':
            self.cfg.device = 'cpu'
        elif self.cfg.device == 'cuda':
            torch.backends.cudnn.deterministic = True
        torch.set_deterministic_debug_mode('warn')

    def _initData(self) -> None:
        # assimilate data config
        data_cfg_path = Path('..', 'simulation', 'configs', f'{self.cfg.d_tag}.yaml')
        assert data_cfg_path.exists(), f'config file {data_cfg_path} does not exist'
        with open(data_cfg_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
            for k, v in self.data_cfg.items():
                setattr(self.cfg, k, v)

        # load validation and test data
        self.dl_valid = self._getDataLoader('valid')
        self.dl_test = self._getDataLoader('test')

    def _getDataLoader(
            self, partition: str, epoch: int = 0, batch_size: int | None = None,
    ) -> Dataloader:
        data_fname = datasetFilename(self.data_cfg, partition, epoch)
        data_path = Path('..', 'outputs', 'data', data_fname)
        return Dataloader(data_path, batch_size=batch_size)

    def _initModel(self) -> None:
        # load model config
        model_cfg_path = Path('..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
        self.model_cfg = modelFromYaml(model_cfg_path,
                                  d_ffx=self.cfg.max_d,
                                  d_rfx=self.cfg.max_q)

        # init model
        self.model = Approximator(self.model_cfg).to(self.device)
        if self.cfg.compile and self.device.type != 'mps':
            self.model.compile()

        # init optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(), lr=self.cfg.lr)

    def save(self, epoch: int = 0, prefix: str = 'latest') -> None:
        fname = checkpointFilename(vars(self.cfg), prefix=prefix)
        path = Path(self.ckpt_dir, fname)
        payload = {
            'timestamp': self.timestamp,
            'epoch': epoch,
            'best_epoch': self.best_epoch,
            'best_valid': self.best_valid,
            'trainer_cfg': vars(self.cfg).copy(),
            'data_cfg': self.data_cfg.copy(),
            'model_cfg': self.model_cfg.to_dict(),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            }
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        logger.info(f'Saved checkpoint to {path}')

    def load(self, prefix: str = 'latest') -> int:
        fname = checkpointFilename(vars(self.cfg), prefix=prefix)
        path = Path(self.ckpt_dir, fname)
        assert path.exists(), f'checkpoint not found: {path}'
        payload = torch.load(path, map_location=self.device)

        # compare configs
        if self.data_cfg != payload['data_cfg']:
            logger.warning('data config mismatch between current and checkpoint')
        if self.model_cfg.to_dict() != payload['model_cfg']:
            logger.warning('model config mismatch between current and checkpoint')

        # load states
        self.model.load_state_dict(payload['model_state'])
        self.optimizer.load_state_dict(payload['optimizer_state'])
        self.timestamp = payload['timestamp']
        self.best_epoch = payload['best_epoch']
        self.best_valid = payload['best_valid']
        return int(payload.get('epoch', 0)) # last completed epoch

    @property
    def info(self) -> str:
        precision = {torch.float32: 32.0, torch.float64: 64.0}
        if self.model.dtype in precision:
            p = precision[self.model.dtype]
        else:
            raise ValueError(f'model has unknown dtype {self.model.dtype}')
        return f'''
====================
data tag:   {self.cfg.d_tag}
model tag:  {self.cfg.m_tag}
# params:   {self.model.n_params}
size [mb]:  {self.model.n_params * (p / 8.0) * 1e-6:.3f}
seed:       {self.cfg.seed}
device:     {self.cfg.device}
compiled:   {self.cfg.compile}
lr:         {self.cfg.lr}
batch size: {self.cfg.bs}
===================='''

    def train(self, epoch: int) -> float:
        dl_train = self._getDataLoader('train', epoch, batch_size=self.cfg.bs)
        iterator = tqdm(dl_train, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [T]')
        loss_train = running_sum = 0.0
        self.model.train()
        self.optimizer.train()
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(iterator):
            # get loss
            batch = toDevice(batch, self.device)
            loss = self.model.forward(batch)
            loss = loss['total'].mean()

            # calculate accumulated gradient with clipped norm
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # log loss
            running_sum += loss.item()
            loss_train = running_sum / (i + 1)
            iterator.set_postfix_str(f'NLL: {loss_train:.3f}')
        return float(loss_train)

    @torch.no_grad()
    def valid(self, epoch: int = 0) -> float:
        iterator = tqdm(self.dl_valid, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [V]')
        loss_valid = running_sum = 0.0
        self.model.eval()
        self.optimizer.eval()
        for i, batch in enumerate(iterator):
            batch = toDevice(batch, self.device)
            loss = self.model.forward(batch)
            loss = loss['total'].mean()
            running_sum += loss.item()
            loss_valid = running_sum / (i + 1)
            iterator.set_postfix_str(f'NLL: {loss_valid:.3f}')
        return float(loss_valid)

    @torch.no_grad()
    def test(self, epoch: int = 0) -> None:
        # expects single batch from dl
        iterator = tqdm(self.dl_test, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [S]')
        self.model.eval()
        self.optimizer.eval()
        for batch in iterator:
            batch = toDevice(batch, self.device)
            proposed = self.model.estimate(batch, n_samples=self.cfg.n_samples)

    def go(self) -> None:
        # optionally load previous checkpoint
        start_epoch = 1
        if self.cfg.load_best:
            last_epoch = self.load('best')
            start_epoch = last_epoch + 1
            print(f'Resumed best checkpoint at epoch {last_epoch}.')
        elif self.cfg.load_latest:
            last_epoch = self.load('latest')
            start_epoch = last_epoch + 1
            print(f'Resumed latest checkpoint at epoch {last_epoch}.')

        if not self.cfg.skip_ref:
            print('\nPerformance before training:')
            self.valid(start_epoch-1)
            self.test(start_epoch-1)

        print(f'\nTraining for {self.cfg.max_epochs - start_epoch + 1} epochs...')
        for epoch in range(start_epoch, self.cfg.max_epochs + 1):
            loss_train = self.train(epoch)
            loss_valid = self.valid(epoch)

            # update best validation loss
            if loss_valid < (self.best_valid - 1e-6):
                self.best_valid = loss_valid
                self.best_epoch = epoch
                if self.cfg.save_best:
                    self.save(epoch, 'best')

            # sample on test set
            if epoch % self.cfg.test_interval == 0:
                self.test(epoch)

            # save latest ckpt
            if self.cfg.save_latest:
                self.save(epoch, 'latest')


# =============================================================================
if __name__ == '__main__':
    cfg = setup()
    trainer = Trainer(cfg)
    print(trainer.info)
    trainer.go()

