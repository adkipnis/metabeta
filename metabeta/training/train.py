import argparse
import yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import schedulefree

from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.config import modelFromYaml
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
    parser.add_argument('--d_tag', type=str, default='toy', help='name of data config file')
    parser.add_argument('--m_tag', type=str, default='toy', help='name of model config file')
    parser.add_argument('-l', '--load', type=int, default=0, help='load model from epoch #l')
    parser.add_argument('--n_samples', type=int, default=200, help='number of samples to draw from posterior on test set')
    parser.add_argument('--compile', action='store_true', help='compile model (default = False)')

    # training & testing
    parser.add_argument('--bs_mini', type=int, default=32, help='number of regression datasets per training minibatch (default = 32)')
    parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate (default = 1e-3)')
    parser.add_argument('--reference', action='store_false', help='do a reference run before training (default = False)')
    parser.add_argument('-e', '--max_epochs', type=int, default=10, help='maximum number of epochs to train (default = 10)')
    parser.add_argument('--test_interval', type=int, default=5, help='sample posterior every #n epochs (default = 5)')
    # parser.add_argument('--patience', type=int, default=20, help='early stopping criterium (default = 20)')

    return parser.parse_args()

# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg

        # reproducibility
        if cfg.reproducible:
            self._reproducible()
        self._seed()

        # misc setup
        self.device = setDevice(self.cfg.device)
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        torch.set_num_threads(cfg.cores)

        # init data
        self._initData()

        # model and optimizer
        self._initModel()
 
    def _reproducible(self) -> None:
        torch.use_deterministic_algorithms(True)
        if self.cfg.device == 'mps':
            self.cfg.device = 'cpu'
        elif self.cfg.device == 'cuda':
            torch.backends.cudnn.deterministic = True

    def _seed(self) -> None:
        s = self.cfg.seed
        np.random.seed(s)
        torch.manual_seed(s)
        if self.cfg.device == 'cuda':
            torch.cuda.manual_seed_all(s)

    def _initData(self) -> None:
        # assimilate data config
        data_cfg_path = Path('..', 'simulation', 'configs', f'{cfg.d_tag}.yaml')
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
        return Dataloader(data_path, batch_size)

    def _initModel(self) -> None:
        # load model config
        model_cfg_path = Path('..', 'models', 'configs', f'{cfg.m_tag}.yaml')
        model_cfg = modelFromYaml(model_cfg_path,
                                  d_ffx=self.cfg.max_d,
                                  d_rfx=self.cfg.max_q)

        # init model
        self.model = Approximator(model_cfg).to(self.device)
        if self.cfg.compile and self.device.type != 'mps':
            self.model.compile()

        # init optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(), lr=self.cfg.lr)

    def train(self, epoch: int) -> None:
        dl_train = self._getDataLoader('train', epoch, batch_size=self.cfg.bs_mini)
        iterator = tqdm(dl_train, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [T]')
        running_sum = 0.0
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
            grad_norm = clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # log loss
            running_sum += loss.item()
            loss_train = running_sum / (i + 1)
            iterator.set_postfix_str(f'NLL: {loss_train:.3f}')

    @torch.no_grad()
    def valid(self, epoch: int = 0) -> None:
        iterator = tqdm(self.dl_valid, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [V]')
        running_sum = 0.0
        self.model.eval()
        self.optimizer.eval()
        for i, batch in enumerate(iterator):
            batch = toDevice(batch, self.device)
            loss = self.model.forward(batch)
            loss = loss['total'].mean()
            running_sum += loss.item()
            loss_valid = running_sum / (i + 1)
            iterator.set_postfix_str(f'NLL: {loss_valid:.3f}')



# =============================================================================
if __name__ == '__main__':
    # --- setup training config
 

    # loop
