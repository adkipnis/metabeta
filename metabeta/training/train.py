import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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
from metabeta.utils.families import LIKELIHOOD_FAMILIES
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import (
    modelFromYaml,
    ApproximatorConfig,
    assimilateConfig,
    loadDataConfig,
)
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import EvaluationSummary, Proposal, dictMean
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler, runIS, runSIR
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.plot import (
    plotRecovery,
    plotCoverage,
    plotSBC,
    plotRfxCorrelationRecovery,
)

logger = logging.getLogger('train.py')


def setup() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--name', type=str, default='toy-n', help='load configs/{name}.yaml')
    parser.add_argument('--d_tag', type=str)
    parser.add_argument('--d_tag_valid', type=str)
    parser.add_argument('--m_tag', type=str)

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
        self.current_epoch = 0
        self.best_score = float('inf')
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
            logger.warning(
                'setting device from mps to cpu for reproducibility - to prevent this, set --reproducible to False'
            )
        elif self.cfg.device == 'cuda':
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True

    def _initData(self) -> None:
        # assimilate data config
        self.data_cfg_train = loadDataConfig(self.cfg.d_tag)
        assimilateConfig(self.cfg, self.data_cfg_train)

        # allow overriding validation data tag independently from training
        self.cfg.d_tag_valid = getattr(self.cfg, 'd_tag_valid', self.cfg.d_tag)
        self.data_cfg_valid = loadDataConfig(self.cfg.d_tag_valid)

        # keep legacy attr names for checkpoint compatibility
        self.data_cfg = self.data_cfg_train

        # load validation data
        self.dl_valid = self._getDataLoader('valid')
        # self.dl_test = self._getDataLoader('test')

    def _getDataLoader(
        self, partition: str, epoch: int = 0, batch_size: int | None = None
    ) -> Dataloader:
        data_cfg = self.data_cfg_train if partition == 'train' else self.data_cfg_valid
        data_fname = datasetFilename(data_cfg, partition, epoch)
        data_path = Path(self.dir, '..', 'outputs', 'data', data_fname)
        return Dataloader(data_path, batch_size=batch_size)

    def _trainingDataAvailable(self, start_epoch: int) -> bool:
        for epoch in range(start_epoch, self.cfg.max_epochs + 1):
            data_fname = datasetFilename(self.data_cfg_train, 'train', epoch)
            data_path = Path(self.dir, '..', 'outputs', 'data', data_fname)
            if not data_path.exists():
                logger.warning(f'{data_path} does not exist')
                return False
        return True

    def _initModel(self) -> None:
        if hasattr(self.cfg, 'model_cfg'):
            assert isinstance(self.cfg.model_cfg, ApproximatorConfig), 'wrong model cfg class'
            self.model_cfg = self.cfg.model_cfg
        else:
            # load model config
            model_cfg_path = Path(self.dir, '..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
            self.model_cfg = modelFromYaml(
                model_cfg_path,
                d_ffx=self.cfg.max_d,
                d_rfx=self.cfg.max_q,
                likelihood_family=self.cfg.likelihood_family,
            )

        # init model
        self.model = Approximator(self.model_cfg).to(self.device)
        if self.cfg.compile and self.device.type != 'mps':
            self.model.compile()

        # init optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.cfg.lr)

    def _initWandb(self) -> None:
        output_dir = Path(self.dir, '..', 'outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project='metabeta',
            name=self.run_name,
            config=vars(self.cfg),
            dir=output_dir,
        )
        wandb.config.update({'data_cfg': self.data_cfg, 'model_cfg': self.model_cfg.to_dict()})
        wandb.define_metric('train/loss_step', step_metric='step/global')
        wandb.define_metric('train/loss_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/loss_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/mean_nrmse_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/mean_lcr_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/early_stop_score_epoch', step_metric='step/epoch')

    def close(self) -> None:
        if self.wandb_run is not None:
            wandb.finish()

    def save(self, prefix: str = 'latest') -> None:
        path = Path(self.ckpt_dir, prefix + '.pt')
        payload = {
            'timestamp': self.timestamp,
            'epoch': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'trainer_cfg': vars(self.cfg).copy(),
            'data_cfg': self.data_cfg.copy(),
            'model_cfg': self.model_cfg.to_dict(),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        logger.debug(f'Saved checkpoint to {path}')

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
        self.best_score = payload['best_score']
        if self.stopper is not None:
            self.stopper.best = self.best_score
        return int(payload.get('epoch', 0))  # last completed epoch

    def getTrackingMetrics(self, eval_summary: EvaluationSummary) -> tuple[float, float, float]:
        mean_nrmse = dictMean(eval_summary.nrmse)
        mean_lcr = dictMean(eval_summary.lcr)
        early_stop_score = mean_nrmse + abs(mean_lcr)
        return mean_nrmse, mean_lcr, early_stop_score

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
valid tag:  {self.cfg.d_tag_valid}
model tag:  {self.cfg.m_tag}
likelihood: {LIKELIHOOD_FAMILIES[self.cfg.likelihood_family]}
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
            alpha = (fkl.detach().abs() / (bkl.detach().abs() + 1e-8)).clamp(max=1.0)
            return fkl + 0.05 * alpha * bkl

        else:
            raise ValueError(f'unknown loss type: {mode}')

    def train(self) -> float:
        dl_train = self._getDataLoader('train', self.current_epoch, batch_size=self.cfg.bs)
        iterator = tqdm(
            dl_train,
            desc=f'Epoch {self.current_epoch:02d}/{self.cfg.max_epochs:02d} [T]',
        )
        loss_train = running_sum = 0.0
        self.model.train()
        self.optimizer.train()
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(iterator):
            if i == 106:
                ...
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
    def valid(self) -> float:
        iterator = tqdm(
            self.dl_valid,
            desc=f'Epoch {self.current_epoch:02d}/{self.cfg.max_epochs:02d} [V]',
        )
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
    def sample(self) -> EvaluationSummary:
        # expects single batch from dl
        self.model.eval()
        self.optimizer.eval()
        batch = next(iter(self.dl_valid))
        batch = toDevice(batch, self.device)

        # sample from proposal distribution
        t0 = time.perf_counter()
        if self.cfg.importance and not self.cfg.sir:
            proposal = runIS(self.model, batch, self.cfg)
        elif self.cfg.sir:
            proposal = runSIR(self.model, batch, self.cfg)
        else:
            proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
            if self.cfg.rescale and self.cfg.likelihood_family == 0:
                proposal.rescale(batch['sd_y'])
        t1 = time.perf_counter()

        # post-process
        proposal.tpd = (t1 - t0) / batch['X'].shape[0]  # time per dataset
        proposal.to('cpu')
        batch = toDevice(batch, 'cpu')
        if self.cfg.rescale and self.cfg.likelihood_family == 0:
            batch = rescaleData(batch)

        # get evaluation summary
        eval_summary = getSummary(proposal, batch, likelihood_family=self.cfg.likelihood_family)
        summary_table = summaryTable(eval_summary, self.cfg.likelihood_family)
        logger.info(summary_table)
        if self.cfg.wandb:
            wandb.log(
                {
                    'summary/table': wandb.Html(f'<pre>{summary_table}</pre>'),
                    'step/epoch': self.current_epoch,
                }
            )

        # make plots
        if self.cfg.plot:
            self.plot(proposal, eval_summary, batch)
        return eval_summary

    def plot(
        self,
        proposal: Proposal,
        eval_summary: EvaluationSummary,
        batch: dict[str, torch.Tensor],
    ) -> None:
        show = not self.cfg.wandb
        path_r = plotRecovery(
            eval_summary,
            batch,
            plot_dir=self.plot_dir,
            epoch=self.current_epoch,
            show=show,
        )
        path_c = plotCoverage(
            eval_summary,
            proposal,
            plot_dir=self.plot_dir,
            epoch=self.current_epoch,
            show=show,
        )
        path_s = plotSBC(
            proposal, batch, plot_dir=self.plot_dir, epoch=self.current_epoch, show=show
        )
        # path_rc = None
        # if proposal.q >= 2:
        #     path_rc = plotRfxCorrelationRecovery(
        #         proposal,
        #         batch,
        #         plot_dir=self.plot_dir,
        #         epoch=self.current_epoch,
        #         show=show,
        #     )
        if self.cfg.wandb:
            image_logs = {
                'plot/recovery': wandb.Image(str(path_r)),
                'plot/coverage': wandb.Image(str(path_c)),
                'plot/sbc': wandb.Image(str(path_s)),
                'step/epoch': self.current_epoch,
            }
            # if path_rc is not None:
            #     image_logs['plot/rfx_correlation_recovery'] = wandb.Image(str(path_rc))
            wandb.log(image_logs)

    def go(self) -> None:
        # optionally load previous checkpoint
        if self.cfg.load_best:
            self.current_epoch = self.load('best')
            print(f'Resumed best checkpoint at epoch {self.current_epoch}.')
        elif self.cfg.load_latest:
            self.current_epoch = self.load('latest')
            print(f'Resumed latest checkpoint at epoch {self.current_epoch}.')

        # check if training data is complete
        # assert self._trainingDataAvailable(self.current_epoch + 1), 'training data incomplete'

        # optionally init wandb (after potential loading and reference run)
        if self.cfg.wandb:
            self._initWandb()

        # optionally get performance before (resumed) training
        if not self.cfg.skip_ref:
            print('\nPerformance before training:')
            self.valid()
            self.sample()

        print(f'\nTraining for {self.cfg.max_epochs - self.current_epoch} epochs...')
        for epoch in range(self.current_epoch + 1, self.cfg.max_epochs + 1):
            self.current_epoch = epoch
            loss_train = self.train()
            loss_valid = self.valid()
            mean_nrmse = None
            mean_lcr = None
            early_stop_score = None

            # sample on test set
            if epoch % self.cfg.sample_interval == 0:
                eval_summary = self.sample()
                mean_nrmse, mean_lcr, early_stop_score = self.getTrackingMetrics(eval_summary)

            # log epoch
            if self.wandb_run is not None:
                logs = {
                    'train/loss_epoch': float(loss_train),
                    'valid/loss_epoch': float(loss_valid),
                    'step/epoch': self.current_epoch,
                }
                if mean_nrmse is not None:
                    logs['valid/mean_nrmse_epoch'] = float(mean_nrmse)
                    logs['valid/mean_lcr_epoch'] = float(mean_lcr) # type: ignore
                    logs['valid/early_stop_score_epoch'] = float(early_stop_score) # type: ignore
                wandb.log(logs)

            # save latest ckpt
            if self.cfg.save_latest:
                self.save('latest')

            # update tracked score and optional early stopping on sample epochs
            if early_stop_score is not None:
                if early_stop_score < (self.best_score - 1e-6):
                    self.best_score = early_stop_score
                    self.best_epoch = self.current_epoch
                    if self.cfg.save_best:
                        self.save('best')

                if self.stopper is not None:
                    self.stopper.update(early_stop_score)
                    if self.stopper.stop:
                        logger.info(f'early stopping at epoch {self.current_epoch}.')
                        break


# =============================================================================
if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    trainer = Trainer(cfg)
    logger.info(trainer.info)
    trainer.go()
    trainer.close()
