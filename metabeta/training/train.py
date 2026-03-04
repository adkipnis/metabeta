import yaml
import time
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch
import wandb
import schedulefree

from metabeta.utils.logger import setupLogging
from metabeta.utils.io import setDevice, datasetFilename, runName
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import EvaluationSummary, Proposal, joinProposals
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.plot import plotRecovery, plotCoverage, plotSBC


logger = logging.getLogger('train.py')


def setup() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--name', type=str, default='default', help='load configs/{name}.yaml')

    # runtime
    parser.add_argument('--device', type=str)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)

    # training
    parser.add_argument('-e', '--max_epochs', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--lr', type=float)

    # evaluation
    parser.add_argument('--importance', action=argparse.BooleanOptionalAction)
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)

    # saving & loading
    parser.add_argument('--r_tag', type=str)
    parser.add_argument('--load_latest', action=argparse.BooleanOptionalAction)
    parser.add_argument('--load_best', action=argparse.BooleanOptionalAction)

    # load and override
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'configs' / f'{args.name}.yaml'
    with open(path, 'r') as p:
        cfg = yaml.safe_load(p)
    cfg.update(vars(args))
    return argparse.Namespace(**cfg)


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
            assert (
                self.cfg.n_samples % self.cfg.sir_iter == 0
            ), 'number of samples must be divisible by number of SIR iterations'

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
        self.wandb_run = None
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

    def _initWandb(self) -> None:
        self.wandb_run = wandb.init(
            project='metabeta',
            name=self.run_name,
            config=vars(self.cfg),
        )
        wandb.config.update({'data_cfg': self.data_cfg, 'model_cfg': self.model_cfg.to_dict()})
        wandb.define_metric('train/loss_step', step_metric='step/global')
        wandb.define_metric('train/loss_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/loss_epoch', step_metric='step/epoch')

    def close(self) -> None:
        if self.wandb_run is not None:
            wandb.finish()

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
        m = batch['m']  # number of groups
        mask = batch['mask_m']  # group mask
        if mode in [
            'backward',
        ]:
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

        # mix KL loss
        elif mode == 'mixed':
            fkl = self.loss(batch, summaries, mode='forward')
            bkl = self.loss(batch, summaries, mode='backward')
            return fkl + 0.001 * bkl

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
            if self.wandb_run is not None:
                wandb.log(
                    {
                        'train/loss_step': float(loss_train),
                        'step/global': self.global_step,
                    }
                )
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
    def sample(self, epoch: int = 0) -> EvaluationSummary:
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

        # evaluation summary
        eval_summary = getSummary(proposal, batch)
        eval_summary.tpd = (t1 - t0) / batch['X'].shape[0]  # time per dataset
        print(summaryTable(eval_summary))

        # plots
        if self.cfg.plot:
            path_r = plotRecovery(
                eval_summary, batch, plot_dir=self.plot_dir, epoch=epoch)
            path_c = plotCoverage(
                eval_summary, proposal, plot_dir=self.plot_dir, epoch=epoch)
            path_s = plotSBC(proposal, batch, plot_dir=self.plot_dir, epoch=epoch)
            if self.wandb_run is not None:
                image_logs = {}
                image_logs['plot/recovery'] = wandb.Image(str(path_r))
                image_logs['plot/coverage'] = wandb.Image(str(path_c))
                image_logs['plot/sbc'] = wandb.Image(str(path_s))
                if image_logs:
                    image_logs['step/epoch'] = epoch
                    wandb.log(image_logs)
        return eval_summary

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

        # optionally init wandb (after potential loading and reference run)
        if self.cfg.wandb:
            self._initWandb()

        # optionally get performance before (resumed) training
        if not self.cfg.skip_ref:
            print('\nPerformance before training:')
            self.valid(start_epoch - 1)
            self.sample(start_epoch - 1)

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
            if self.wandb_run is not None:
                wandb.log(
                    {
                        'train/loss_epoch': float(loss_train),
                        'valid/loss_epoch': float(loss_valid),
                        'step/epoch': epoch,
                    },
                )

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
