import yaml
import time
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import schedulefree

from metabeta.evaluation.intervals import expectedCoverageError
from metabeta.utils.logger import setupLogging
from metabeta.utils.io import setDevice, datasetFilename, runName
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import Proposal, joinProposals
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.evaluation.summary import flatSummary, dependentSummary, recoveryPlot

logger = logging.getLogger('train.py')


def setup() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--device', type=str, default='cpu', help='device to use [cpu, cuda, mps], (default = cpu)')
    parser.add_argument('--cores', type=int, default=8, help='number of processor cores to use (default = 8)')
    parser.add_argument('--reproducible', action='store_false', help='use deterministic learning trajectory (default = True)')
    parser.add_argument('--tb', action='store_true', help='enable tensorboard logging (default = False)')
    parser.add_argument('--verbosity', type=int, default=0, help='verbosity level (0: warnings, 1: infos, 2: debug, default=0)')

    # model & optimizer
    parser.add_argument('-m', '--m_tag', type=str, default='toy', help='name of model config file')
    parser.add_argument('--n_samples', type=int, default=512, help='number of samples to draw from posterior on test set')
    parser.add_argument('--compile', action='store_true', help='compile model (default = False)')
    parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate (default = 1e-3)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip grad norm to this value (default = 1.0)')
    parser.add_argument('--loss_type', type=str, default='forward+', help='KL loss type [forward, backward, mixed] (default = mixed)')

    # data
    parser.add_argument('-d', '--d_tag', type=str, default='toy', help='name of data config file')
    parser.add_argument('--bs', type=int, default=32, help='number of regression datasets per training minibatch (default = 32)')

    # steps
    parser.add_argument('--skip_ref', action='store_true', help='skip the reference run before training (default = False)')
    parser.add_argument('-e', '--max_epochs', type=int, default=10, help='maximum number of epochs to train (default = 10)')
    parser.add_argument('--sample_interval', type=int, default=5, help='sample posterior every #n epochs (default = 5)')
    parser.add_argument('--patience', type=int, default=5, help='early stopping criterion (default = 10)')

    # evaluation
    parser.add_argument('--rescale', action='store_false', help='use original scale of y for evaluation (default = True)')
    parser.add_argument('--importance', action='store_true', help='use importance sampling before evaluation (default = False)')
    parser.add_argument('--sir', action='store_true', help='use sampling importance resampling before evaluation (default = False)')
    parser.add_argument('--sir_iter', type=int, default=8, help='number of SIR iterations (default = 10)')
    parser.add_argument('--plot', action='store_false', help='plot evaluation results (default = True)')

    # saving & loading
    parser.add_argument('--r_tag', type=str, default='', help='run tag (default="")')
    parser.add_argument('--save_latest', action='store_false', help='save latest model after each epoch (default = True)')
    parser.add_argument('--save_best', action='store_true', help='track and save best model wrt. validation set (default = True)')
    parser.add_argument('--load_latest', action='store_true', help='load latest model before training (default = False)')
    parser.add_argument('--load_best', action='store_true', help='load best model wrt. validation set (overwrites load_latest, default = False)')

    return parser.parse_args()


# -----------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 1e-3) -> None:
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
            if self.counter >= self.patience:
                self.stop = True


# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.dir = Path(__file__).resolve().parent

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

        # check IS sizes
        if self.cfg.sir:
            assert self.cfg.n_samples % self.cfg.sir_iter == 0, \
                'number of samples must be divisible by number of SIR iterations'

        # checkpoint dir
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_name = runName(vars(self.cfg))
        self.ckpt_dir = Path(self.dir, '..', 'outputs', 'checkpoints', self.run_name)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # plot dir
        self.plot_dir = None
        if self.cfg.plot:
            self.plot_dir = Path(self.dir, '..', 'outputs', 'plots', self.run_name)
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        # tracking & logging
        self.best_valid = float('inf')
        self.best_epoch = 0
        self.global_step = 0
        self.writer = None
        self.stopper = None
        if self.cfg.patience > 0:
            self.stopper = EarlyStopping(self.cfg.patience)
            if not self.cfg.save_best:
                logger.warning('early stopping enabled without saving best checkpoints!')

    def _reproducible(self) -> None:
        torch.use_deterministic_algorithms(True)
        if self.cfg.device == 'mps':
            self.cfg.device = 'cpu'
        elif self.cfg.device == 'cuda':
            torch.backends.cudnn.deterministic = True
        torch.set_deterministic_debug_mode('warn')

    def _initData(self) -> None:
        # assimilate data config
        data_cfg_path = Path(self.dir, '..', 'simulation', 'configs', f'{self.cfg.d_tag}.yaml')
        assert data_cfg_path.exists(), f'config file {data_cfg_path} does not exist'
        with open(data_cfg_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
            for k, v in self.data_cfg.items():
                setattr(self.cfg, k, v)

        # load validation and test data
        self.dl_valid = self._getDataLoader('valid')
        self.dl_test = self._getDataLoader('test')

    def _getDataLoader(
        self, partition: str, epoch: int = 0, batch_size: int | None = None
    ) -> Dataloader:
        data_fname = datasetFilename(self.data_cfg, partition, epoch)
        data_path = Path(self.dir, '..', 'outputs', 'data', data_fname)
        return Dataloader(data_path, batch_size=batch_size)

    def _initModel(self) -> None:
        # load model config
        model_cfg_path = Path(self.dir, '..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
        self.model_cfg = modelFromYaml(model_cfg_path, d_ffx=self.cfg.max_d, d_rfx=self.cfg.max_q)

        # init model
        self.model = Approximator(self.model_cfg).to(self.device)
        if self.cfg.compile and self.device.type != 'mps':
            self.model.compile()

        # init optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.cfg.lr)

    def _initWriter(self) -> None:
        self.tb_path = Path(
            self.dir, '..', 'outputs', 'tensorboard', self.run_name + '_' + self.timestamp)
        self.tb_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_path))

        # log configs once
        self.writer.add_text('cfg/trainer', yaml.safe_dump(vars(self.cfg), sort_keys=True), 0)
        self.writer.add_text('cfg/data', yaml.safe_dump(self.data_cfg, sort_keys=True), 0)
        self.writer.add_text(
            'cfg/model', yaml.safe_dump(self.model_cfg.to_dict(), sort_keys=True), 0
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def save(self, epoch: int = 0, prefix: str = 'latest') -> None:
        path = Path(self.ckpt_dir, prefix + '.pt')
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
        path = Path(self.ckpt_dir, prefix + '.pt')
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
        if self.stopper is not None:
            self.stopper.best = self.best_valid
        return int(payload.get('epoch', 0))  # last completed epoch

    @property
    def info(self) -> str:
        precision = {torch.float32: 32.0, torch.float64: 64.0}
        if self.model.dtype in precision:
            p = precision[self.model.dtype]
        else:
            raise ValueError(f'model has unknown dtype {self.model.dtype}')
        return f"""
====================
data tag:   {self.cfg.d_tag}
model tag:  {self.cfg.m_tag}
# params:   {self.model.n_params}
size [mb]:  {self.model.n_params * (p / 8.0) * 1e-6:.3f}
seed:       {self.cfg.seed}
device:     {self.cfg.device}
compiled:   {self.cfg.compile}
loss type:  {self.cfg.loss_type}
lr:         {self.cfg.lr}
batch size: {self.cfg.bs}
===================="""

    def loss(
            self,
            batch: dict[str, torch.Tensor],
            summaries: tuple[torch.Tensor, torch.Tensor] | None = None,
            mode: str = '',
    ) -> torch.Tensor:
        if not mode:
            mode = self.cfg.loss_type

        #  init group variables
        m = batch['m'] # number of groups
        mask = batch['mask_m'] # group mask
        if mode in ['backward', 'coverage']:
            m = m.unsqueeze(-1)
            mask = mask.unsqueeze(-1)

        # precompute summaries
        if summaries is None:
            summaries = self.model.summarize(batch)

        # forward KL loss
        if mode == 'forward':
            log_probs = self.model.forward(batch, summaries)
            lq_g = log_probs['global']
            lq_l = log_probs['local'] * mask
            lq = lq_g + lq_l.sum(1) / m
            return -lq.mean()

        # backward KL loss
        elif mode == 'backward':
            proposal = self.model.backward(batch, summaries, n_samples=32)
            lq_g, lq_l = proposal.log_probs
            lq_l = lq_l * mask
            lq = lq_g + lq_l.sum(1) / m
            ll, lp = ImportanceSampler(batch).unnormalizedPosterior(proposal)
            return (lq - lp - ll).mean()

        # coverage loss
        elif mode == 'coverage':
            proposal = self.model.backward(batch, summaries, n_samples=32)
            ece_dict = expectedCoverageError(proposal, batch)
            ece = 100.0 * sum(ece_dict.values()) / len(ece_dict)
            esce = ece.square()
            return esce

        # mix KL loss
        elif mode == 'mixed':
            fkl = self.loss(batch, summaries, mode='forward')
            bkl = self.loss(batch, summaries, mode='backward')
            return fkl + 0.001 * bkl

        # forward KL loss + regularized
        elif mode == 'forward+':
            fkl = self.loss(batch, summaries, mode='forward')
            esce = self.loss(batch, summaries, mode='coverage')
            return fkl + 0.1 * esce

        else:
            raise ValueError(f'unknown loss type: {mode}')

    def train(self, epoch: int) -> float:
        dl_train = self._getDataLoader('train', epoch, batch_size=self.cfg.bs)
        iterator = tqdm(dl_train, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [T]')
        loss_train = running_sum = 0.0
        self.model.train()
        self.optimizer.train()
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(iterator):
            batch = toDevice(batch, self.device)
            loss = self.loss(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
            if torch.isfinite(grad_norm):
                self.optimizer.step()
                self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)

            # write loss
            running_sum += loss.item()
            loss_train = running_sum / (i + 1)
            iterator.set_postfix_str(f'Loss: {loss_train:.3f}')
            if self.writer is not None:
                self.writer.add_scalar('train/loss_step', float(loss_train), self.global_step)
        return float(loss_train)

    @torch.inference_mode()
    def valid(self, epoch: int = 0) -> float:
        iterator = tqdm(self.dl_valid, desc=f'Epoch {epoch:02d}/{self.cfg.max_epochs:02d} [V]')
        loss_valid = running_sum = 0.0
        self.model.eval()
        self.optimizer.eval()
        for i, batch in enumerate(iterator):
            batch = toDevice(batch, self.device)
            loss = self.loss(batch)
            running_sum += loss.item()
            loss_valid = running_sum / (i + 1)
            iterator.set_postfix_str(f'Loss: {loss_valid:.3f}')
        return float(loss_valid)

    @torch.inference_mode()
    def sample(self, epoch: int = 0) -> None:
        # expects single batch from dl
        self.model.eval()
        self.optimizer.eval()
        batch = next(iter(self.dl_valid))
        batch = toDevice(batch, self.device)

        # get proposal distribution
        t0 = time.perf_counter()
        if not self.cfg.sir:
            proposal, batch = self._sampleSingle(batch)
        else:
            proposal, batch = self._sampleMulti(batch)            
        t1 = time.perf_counter()
        tpd = (t1 - t0) / batch['X'].shape[0]  # time per dataset

        # evaluation summary
        print(dependentSummary(proposal, batch))
        print(flatSummary(proposal, batch, tpd=tpd))
        if cfg.plot:
            recoveryPlot(proposal, batch, plot_dir=self.plot_dir, epoch=epoch)

    def _sampleSingle(
            self, batch: dict[str, torch.Tensor]
    ) -> tuple[Proposal, dict[str, torch.Tensor]]:
        proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)

        # unnormalize proposal and batch
        if self.cfg.rescale:
            proposal.rescale(batch['sd_y'])
            batch = rescaleData(batch)

        # importance weighing
        if self.cfg.importance:
            imp_sampler = ImportanceSampler(batch, sir=False)
            proposal = imp_sampler(proposal)
        return proposal, batch

    def _sampleMulti(
            self, batch: dict[str, torch.Tensor]
    ) -> tuple[Proposal, dict[str, torch.Tensor]]:
        # separate normalized and unnormalized batch
        eval_batch = batch
        if self.cfg.rescale:
            eval_batch = rescaleData(batch)
        n_sir = self.cfg.n_samples // self.cfg.sir_iter
        imp_sampler = ImportanceSampler(eval_batch, sir=True, n_sir=n_sir)
        selected = []
        n_remaining = self.cfg.n_samples

        # Sampling Importance Resampling (SIR)
        while n_remaining > 0:
            proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
            if self.cfg.rescale:
                proposal.rescale(batch['sd_y'])
            proposal = imp_sampler(proposal)
            selected.append(proposal)
            n_remaining -= proposal.n_samples
        proposal = joinProposals(selected)
        return proposal, eval_batch

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

        # optionally get performance before (resumed) training
        if not self.cfg.skip_ref:
            print('\nPerformance before training:')
            self.valid(start_epoch - 1)
            self.sample(start_epoch - 1)

        # optionally init tensorboard (after potential loading and reference run)
        if self.cfg.tb:
            self._initWriter()

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
            if epoch % self.cfg.sample_interval == 0:
                self.sample(epoch)

            # log epoch
            if self.writer is not None:
                self.writer.add_scalar('train/loss_epoch', float(loss_train), epoch)
                self.writer.add_scalar('valid/loss_epoch', float(loss_valid), epoch)

            # save latest ckpt
            if self.cfg.save_latest:
                self.save(epoch, 'latest')

            # optional early stopping
            if self.stopper is not None:
                self.stopper.update(loss_valid)
                if self.stopper.stop:
                    self.sample(epoch)
                    logger.info(f'early stopping at epoch {epoch}.')
                    break


# =============================================================================
if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    trainer = Trainer(cfg)
    print(trainer.info)
    trainer.go()
    trainer.close()
