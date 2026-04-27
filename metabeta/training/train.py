import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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
from metabeta.utils.templates import (
    setupConfigParser,
    generateTrainingConfig,
    saveConfigToCheckpoint,
    CLI_ONLY_PARAMS,
)
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import (
    EvaluationSummary,
    Proposal,
    dictMean,
    concatProposalsBatch,
)
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler, runIS, runSIR
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.evaluation.intervals import RFX_COVERAGE_AGGREGATIONS
from metabeta.plotting import (
    plotRecovery,
    plotCoverage,
    plotSBC,
    # plotRfxCorrelationRecovery,
)

logger = logging.getLogger('train.py')


# fmt: off
def setup() -> argparse.Namespace:
    """Parse command line arguments.

    Usage modes
    -----------
    Fresh start (template-based):
        python train.py --size small --family 0 --ds_type toy

        Generates config from size/family/ds_type presets. The checkpoint
        directory name is derived from the config (e.g. normal_dsmall-n-toy_msmall_s42).
        Model architecture is loaded from configs/models/{model_id}.yaml, which
        defaults to the same value as --size but can be overridden with --model_id.

    Continue from checkpoint config:
        python train.py --config path/to/checkpoint/config.yaml --load_latest

        Loads the full config from a saved YAML. Explicit CLI args (e.g. --max_epochs)
        override the YAML values. Use --load_latest to resume from the most recent
        saved weights, or --load_best to resume from the best validation checkpoint.

    Override individual fields:
        python train.py --size tiny --family 0 --ds_type toy --lr 1e-4 --bs 64

        Any training hyperparameter can be overridden on top of both template-based
        configs and loaded YAML configs.

    Decouple model and data size:
        python train.py --size small --model_id large

        Uses small-size data dimensions but loads the large model architecture.
    """
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        epilog='Advanced options (max_grad_norm, sir, sir_iter, rescale, skip_ref, save_latest, save_best) can be set via --config.',
    )

    # Template-based config generation (primary interface)
    parser.add_argument('--size', type=str, default='small', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='mixed', help='Training dataset type: toy|flat|scm|mixed|sampled|observed')
    parser.add_argument('--valid_ds_type', type=str, default='sampled', help='Validation dataset type: toy|flat|scm|mixed|sampled|observed')

    # Alternative: load config from a saved YAML (e.g. a checkpoint config.yaml)
    parser.add_argument('--config', type=str, help='Path to a saved config.yaml; explicit CLI args override its values')
    parser.add_argument('--data_id', type=str, help='Training dataset ID (subfolder name under outputs/data/)')
    parser.add_argument('--data_id_valid', type=str, help='Validation dataset ID (defaults to data_id with valid_ds_type suffix)')
    parser.add_argument('--model_id', type=str, help='Model architecture ID; loads configs/models/{model_id}.yaml (defaults to --size)')

    # CLI-only runtime params (never written to config.yaml)
    parser.add_argument('--device', type=str, default='cpu', help='Compute device: cpu|cuda')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False, help='Log metrics to Weights & Biases')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--verbosity', type=int, default=1, help='Logging verbosity level')

    # Training hyperparameters (override template or loaded YAML)
    parser.add_argument('-e', '--max_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--bs', type=int, help='Batch size (number of datasets per step, default = 32)')
    parser.add_argument('--accum_steps', type=int, help='Gradient accumulation steps; effective batch size = bs × accum_steps (default = 1)')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--loss_type', type=str, help='Loss type: forward|backward|mixed (default = forward)')
    parser.add_argument('--n_samples', type=int, help='Posterior samples drawn per evaluation dataset (default = 512)')
    parser.add_argument('--patience', type=int, help='Early stopping patience in epochs; 0 = disabled (default = 0)')
    parser.add_argument('--sample_interval', type=int, help='Run full posterior evaluation every N epochs (default = 20)')
    parser.add_argument('--cores', type=int, help='CPU thread count passed to torch.set_num_threads (default = 8)')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help='Compile model with torch.compile (default = False)')
    parser.add_argument('--reproducible', action=argparse.BooleanOptionalAction, help='Enable deterministic algorithms for reproducibility (default = True)')

    # Evaluation settings
    parser.add_argument('--importance', action=argparse.BooleanOptionalAction, help='Run importance sampling evaluation')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Generate evaluation plots after each epoch')
    parser.add_argument(
        '--rfx_coverage_aggregation',
        type=str,
        choices=RFX_COVERAGE_AGGREGATIONS,
        help='RFX coverage aggregation: group_weighted (default) or slot_mean (legacy)',
    )

    # Saving & loading
    parser.add_argument('--r_tag', type=str, help='Run tag suffix appended to the checkpoint directory name')
    parser.add_argument('--load_latest', action=argparse.BooleanOptionalAction, help='Resume training from latest.pt in the checkpoint directory')
    parser.add_argument('--load_best', action=argparse.BooleanOptionalAction, help='Resume training from best.pt in the checkpoint directory')

    return setupConfigParser(parser, generateTrainingConfig, 'Train neural approximators.')
# fmt: on


# -----------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 0, delta: float = 1e-3) -> None:
        self.patience = patience
        self.delta = delta
        self.best_nrmse = float('inf')
        self.best_median_nll = float('inf')
        self.counter = 0
        self.stop = False

    def update(self, mean_nrmse: float, median_nll: float) -> bool:
        improved_nrmse = (self.best_nrmse - mean_nrmse) > self.delta
        improved_nll = (self.best_median_nll - median_nll) > self.delta
        improved = improved_nrmse or improved_nll

        if improved_nrmse:
            self.best_nrmse = mean_nrmse
        if improved_nll:
            self.best_median_nll = median_nll

        if improved:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return improved


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

        # Save resolved config to checkpoint dir for reproducibility
        saveConfigToCheckpoint(vars(self.cfg), self.ckpt_dir)

        # plot dir
        self.plot_dir = None
        if getattr(self.cfg, 'plot', False):
            self.plot_dir = Path(self.dir, '..', 'outputs', 'plots', self.run_name)
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        # tracking & logging
        self.current_epoch = 0
        self.best_nrmse = float('inf')
        self.best_median_nll = float('inf')
        self.best_epoch = 0
        self.global_step = 0
        self.wandb_run = None
        self.wandb_run_id = None
        self.stopper = None
        if self.cfg.patience > 0:
            self.stopper = EarlyStopping(self.cfg.patience)
            if not getattr(self.cfg, 'save_best', True):
                logger.warning('early stopping enabled without saving best checkpoints!')

    def _reproducible(self) -> None:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    def _initData(self) -> None:
        # assimilate data config
        self.data_cfg_train = loadDataConfig(self.cfg.data_id)
        assimilateConfig(self.cfg, self.data_cfg_train)

        # allow overriding validation data id independently from training
        self.cfg.data_id_valid = getattr(self.cfg, 'data_id_valid', self.cfg.data_id)
        self.data_cfg_valid = loadDataConfig(self.cfg.data_id_valid)

        # keep legacy attr names for checkpoint compatibility
        self.data_cfg = self.data_cfg_train

        # load validation data
        self.dl_valid = self._getDataLoader('valid', batch_size=8)
        # self.dl_test = self._getDataLoader('test')

    def _getDataLoader(
        self, partition: str, epoch: int = 0, batch_size: int | None = None
    ) -> Dataloader:
        data_cfg = self.data_cfg_train if partition == 'train' else self.data_cfg_valid
        data_fname = datasetFilename(partition, epoch)
        data_subdir = data_cfg['data_id']
        data_path = Path(self.dir, '..', 'outputs', 'data', data_subdir, data_fname)
        sortish = batch_size is not None
        return Dataloader(
            data_path,
            batch_size=batch_size,
            sortish=sortish,
            shuffle=partition == 'train',
            bucket_mult=50,
            sort_seed=epoch,
            max_d=data_cfg.get('max_d'),
            max_q=data_cfg.get('max_q'),
            # num_workers=0,
            # persistent_workers=(partition != 'train'),
        )

    def _trainingDataAvailable(self, start_epoch: int) -> bool:
        for epoch in range(start_epoch, self.cfg.max_epochs + 1):
            data_fname = datasetFilename('train', epoch)
            data_subdir = self.data_cfg_train['data_id']
            data_path = Path(self.dir, '..', 'outputs', 'data', data_subdir, data_fname)
            if not data_path.exists():
                logger.warning(f'{data_path} does not exist')
                return False
        return True

    def _initModel(self) -> None:
        if hasattr(self.cfg, 'model_cfg'):
            assert isinstance(self.cfg.model_cfg, ApproximatorConfig), 'wrong model cfg class'
            self.model_cfg = self.cfg.model_cfg
        else:
            # load model config from new location
            model_cfg_path = Path(self.dir, '..', 'configs', 'models', f'{self.cfg.model_id}.yaml')
            self.model_cfg = modelFromYaml(
                model_cfg_path,
                d_ffx=self.cfg.max_d,
                d_rfx=self.cfg.max_q,
                likelihood_family=self.cfg.likelihood_family,
            )

        if analytical_context := getattr(self.cfg, 'analytical_context', None):
            self.model_cfg = self.model_cfg.model_copy(
                update={'analytical_context': analytical_context}
            )

        # init model
        self.model = Approximator(self.model_cfg).to(self.device)
        if self.cfg.compile and self.device.type == 'cuda':
            self.model.compile()

        if not hasattr(self.cfg, 'rfx_coverage_aggregation'):
            self.cfg.rfx_coverage_aggregation = 'group_weighted'

        # init optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.cfg.lr)

    def _initWandb(self) -> None:
        output_dir = Path(self.dir, '..', 'outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = dict(
            project='metabeta',
            name=self.run_name,
            config=vars(self.cfg),
            dir=output_dir,
            id=self.wandb_run_id,
            resume='must' if self.wandb_run_id is not None else None,
        )
        try:
            self.wandb_run = wandb.init(**init_kwargs)
        except wandb.errors.UsageError:
            print(
                f'WARNING: Could not resume wandb run {self.wandb_run_id!r}; starting a new run.'
            )
            init_kwargs.update(id=None, resume=None)
            self.wandb_run = wandb.init(**init_kwargs)
        wandb.config.update({'data_cfg': self.data_cfg, 'model_cfg': self.model_cfg.to_dict()})
        wandb.define_metric('train/loss_step', step_metric='step/global')
        wandb.define_metric('train/grad_norm', step_metric='step/global')
        wandb.define_metric('train/loss_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/loss_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/mean_nrmse_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/mean_abs_lcr_epoch', step_metric='step/epoch')
        wandb.define_metric('valid/median_nll_epoch', step_metric='step/epoch')

    def close(self) -> None:
        if self.wandb_run is not None:
            wandb.finish()

    def save(self, prefix: str = 'latest') -> None:
        path = Path(self.ckpt_dir, prefix + '.pt')
        payload = {
            'timestamp': self.timestamp,
            'epoch': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_nrmse': self.best_nrmse,
            'best_median_nll': self.best_median_nll,
            'trainer_cfg': {k: v for k, v in vars(self.cfg).items() if k not in CLI_ONLY_PARAMS},
            'data_cfg': self.data_cfg.copy(),
            'model_cfg': self.model_cfg.to_dict(),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'wandb_run_id': self.wandb_run.id if self.wandb_run is not None else None,
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
        self.best_nrmse = payload['best_nrmse']
        self.best_median_nll = payload['best_median_nll']
        self.wandb_run_id = payload.get('wandb_run_id')
        if self.stopper is not None:
            self.stopper.best_nrmse = self.best_nrmse
            self.stopper.best_median_nll = self.best_median_nll
        return int(payload.get('epoch', 0))  # last completed epoch

    def getTrackingMetrics(self, eval_summary: EvaluationSummary) -> tuple[float, float, float]:
        mean_nrmse = dictMean(eval_summary.nrmse)
        mean_abs_lcr = dictMean(eval_summary.abs_lcr)
        median_nll = eval_summary.mloonll if eval_summary.mloonll else 0.0
        return mean_nrmse, mean_abs_lcr, median_nll

    @property
    def info(self) -> str:
        precision = {torch.float32: 32.0, torch.float64: 64.0}
        if self.model.dtype in precision:
            p = precision[self.model.dtype]
        else:
            raise ValueError(f'model has unknown dtype {self.model.dtype}')
        return f"""
====================
data id:    {self.cfg.data_id}
valid id:   {self.cfg.data_id_valid}
model id:   {self.cfg.model_id}
likelihood: {LIKELIHOOD_FAMILIES[self.cfg.likelihood_family]}
# params:   {self.model.n_params}
size [mb]:  {self.model.n_params * (p / 8.0) * 1e-6:.3f}
seed:       {self.cfg.seed}
device:     {self.cfg.device}
compiled:   {self.cfg.compile and self.model.device.type == 'cuda'}
loss type:  {self.cfg.loss_type}
lr:         {self.cfg.lr}
batch size: {self.cfg.bs}{f' × {self.cfg.accum_steps} = {self.cfg.bs * self.cfg.accum_steps} (effective)' if self.cfg.accum_steps > 1 else ''}
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
        n_batches = len(dl_train)
        accum_steps = self.cfg.accum_steps
        iterator = tqdm(
            dl_train,
            desc=f'Epoch {self.current_epoch:02d}/{self.cfg.max_epochs:02d} [T]',
        )
        loss_train = running_sum = 0.0
        self.model.train()
        self.optimizer.train()
        self.optimizer.zero_grad(set_to_none=True)
        # size of the final (possibly partial) accumulation window
        trailing = n_batches % accum_steps or accum_steps  # accum_steps when evenly divisible
        for i, batch in enumerate(iterator):
            batch = toDevice(batch, self.device)
            is_step = (i + 1) % accum_steps == 0 or i == n_batches - 1
            # trailing window: correct divisor so its effective lr matches full windows
            in_trailing = i >= n_batches - trailing
            window_size = trailing if in_trailing else accum_steps
            raw_loss = self.loss(batch)
            (raw_loss / window_size).backward()

            # write loss (track unscaled value so it's comparable to valid loss)
            running_sum += raw_loss.item()
            loss_train = running_sum / (i + 1)
            iterator.set_postfix_str(f'Loss: {loss_train:.3f}')

            # optimizer step at accumulation boundary or end of epoch
            if is_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                    self.global_step += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.wandb_run is not None:
                    wandb.log(
                        {
                            'train/loss_step': float(loss_train),
                            'train/grad_norm': float(grad_norm),
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
        loss_valid = total_weighted_loss = 0.0
        total_count = 0
        self.model.eval()
        self.optimizer.eval()
        for i, batch in enumerate(iterator):
            batch = toDevice(batch, self.device)
            loss = self.loss(batch)
            batch_size = batch['X'].shape[0]
            total_weighted_loss += loss.item() * batch_size
            total_count += batch_size
            loss_valid = total_weighted_loss / max(total_count, 1)
            iterator.set_postfix_str(f'Loss: {loss_valid:.3f}')
        return float(loss_valid)

    @torch.inference_mode()
    def sample(self) -> EvaluationSummary:
        iterator = tqdm(
            self.dl_valid,
            desc=f'Epoch {self.current_epoch:02d}/{self.cfg.max_epochs:02d} [S]',
        )
        # batch = next(iter(self.dl_valid_full))
        self.model.eval()
        self.optimizer.eval()
        proposals = []
        n_datasets = 0

        # sample from proposal distribution over all validation minibatches
        t0 = time.perf_counter()
        for batch in iterator:
            batch = toDevice(batch, self.device)
            if self.cfg.importance and not self.cfg.sir:
                proposal = runIS(self.model, batch, self.cfg)
            elif self.cfg.sir:
                proposal = runSIR(self.model, batch, self.cfg)
            else:
                proposal = self.model.estimate(batch, n_samples=self.cfg.n_samples)
                if self.cfg.rescale and self.cfg.likelihood_family == 0:
                    proposal.rescale(batch['sd_y'])
            proposal.to('cpu')
            batch = toDevice(batch, 'cpu')
            if self.cfg.rescale and self.cfg.likelihood_family == 0:
                batch = rescaleData(batch)
            proposals.append(proposal)
            n_datasets += batch['X'].shape[0]
        t1 = time.perf_counter()

        # merge proposals over minibatches, but evaluate on canonical full-batch collate
        proposal = concatProposalsBatch(proposals)
        batch = self.dl_valid.fullBatch()
        batch = toDevice(batch, 'cpu')
        if self.cfg.rescale and self.cfg.likelihood_family == 0:
            batch = rescaleData(batch)

        # post-process
        proposal.tpd = (t1 - t0) / max(n_datasets, 1)  # time per dataset

        # get evaluation summary
        eval_summary = getSummary(
            proposal,
            batch,
            likelihood_family=self.cfg.likelihood_family,
            rfx_coverage_aggregation=self.cfg.rfx_coverage_aggregation,
        )
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
        if getattr(self.cfg, 'plot', False):
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
        if getattr(self.cfg, 'load_best', False):
            self.current_epoch = self.load('best')
            print(f'Resumed best checkpoint at epoch {self.current_epoch}.')
        elif getattr(self.cfg, 'load_latest', False):
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
            mean_abs_lcr = None
            median_nll = None

            # sample on test set
            if epoch % self.cfg.sample_interval == 0:
                eval_summary = self.sample()
                mean_nrmse, mean_abs_lcr, median_nll = self.getTrackingMetrics(eval_summary)

            # log epoch
            if self.wandb_run is not None:
                logs = {
                    'train/loss_epoch': float(loss_train),
                    'valid/loss_epoch': float(loss_valid),
                    'step/epoch': self.current_epoch,
                }
                if mean_nrmse is not None:
                    logs['valid/mean_nrmse_epoch'] = float(mean_nrmse)
                    logs['valid/mean_abs_lcr_epoch'] = float(mean_abs_lcr)  # type: ignore
                    logs['valid/median_nll_epoch'] = float(median_nll)  # type: ignore
                wandb.log(logs)

            # save latest ckpt
            if getattr(self.cfg, 'save_latest', True):
                self.save('latest')

            # update tracked metrics and optional early stopping on sample epochs
            if mean_nrmse is not None:
                improved_nrmse = mean_nrmse < (self.best_nrmse - 1e-6)
                improved_nll = median_nll < (self.best_median_nll - 1e-6)  # type: ignore
                improved = improved_nrmse or improved_nll

                if improved_nrmse:
                    self.best_nrmse = mean_nrmse
                if improved_nll:
                    self.best_median_nll = median_nll  # type: ignore
                if improved:
                    self.best_epoch = self.current_epoch
                    if getattr(self.cfg, 'save_best', True):
                        self.save('best')

                if self.stopper is not None:
                    self.stopper.update(float(mean_nrmse), float(median_nll))  # type: ignore
                    if self.stopper.stop:
                        logger.info(f'early stopping at epoch {self.current_epoch}.')
                        break


# =============================================================================
def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    trainer = Trainer(cfg)
    logger.info(trainer.info)
    trainer.go()
    trainer.close()


if __name__ == '__main__':
    main()
