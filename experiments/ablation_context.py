"""Context ablation study: compare analytical conditioning strategies for 15 epochs.

Variants
--------
full          Current baseline: beta_est (clamped), sigma_rfx_est (log1p),
              blup_est (clamped), blup_std (log1p).
no_log1p      Same stats, raw-clamped (no log1p on sigma_rfx / blup_std).
no_blup_std   Drop blup variance; keep beta_est, sigma_rfx_est, blup_est.
no_analytics  Only priors + family encoding + n_obs; no GLMM-derived stats.

Usage (from metabeta/training/)
--------------------------------
uv run python ablation.py
uv run python ablation.py --max_epochs 10 --bs 32 --device cpu
"""

from __future__ import annotations

import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import logging
from pathlib import Path

import torch
from tabulate import tabulate

from metabeta.utils.logger import setupLogging
from metabeta.utils.io import setDevice
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import modelFromYaml, loadDataConfig, assimilateConfig
from metabeta.utils.templates import generateTrainingConfig
from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.utils.evaluation import EvaluationSummary, dictMean, concatProposalsBatch

# reuse Trainer machinery without duplicating it
from metabeta.training.train import Trainer, setup as _train_setup  # noqa: F401 — used via Trainer

logger = logging.getLogger('ablation.py')

VARIANTS: list[str] = ['full', 'no_log1p', 'no_blup_std', 'no_analytics']

# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description='Run context ablation study across analytical_context variants.',
    )
    parser.add_argument('--size',          type=str, default='small',   help='Size preset')
    parser.add_argument('--family',        type=int, default=0,         help='Likelihood family')
    parser.add_argument('--ds_type',       type=str, default='mixed',   help='Training dataset type')
    parser.add_argument('--valid_ds_type', type=str, default='sampled', help='Validation dataset type')
    parser.add_argument('--device',        type=str, default='cpu',     help='Compute device')
    parser.add_argument('--seed',          type=int, default=42,        help='Random seed')
    parser.add_argument('--verbosity',     type=int, default=1,         help='Logging verbosity')
    parser.add_argument('-e', '--max_epochs', type=int, default=15,     help='Training epochs per variant')
    parser.add_argument('--bs',            type=int, default=32,        help='Batch size')
    parser.add_argument('--lr',            type=float, default=1e-3,    help='Learning rate')
    parser.add_argument('--n_samples',     type=int, default=256,       help='Posterior samples for evaluation')
    parser.add_argument('--cores',         type=int, default=8,         help='CPU threads')
    parser.add_argument('--variants', nargs='+', default=VARIANTS,
                        help='Subset of variants to run (default: all four)')
    args = parser.parse_args()
    return args
# fmt: on


def _build_trainer_cfg(args: argparse.Namespace, variant: str) -> argparse.Namespace:
    """Build a Trainer-compatible namespace for one variant."""
    import copy

    cfg = copy.copy(args)

    # Generate base training config from template (sets data_id, model_id, hyperparams, etc.)
    template = generateTrainingConfig(
        size=args.size,
        family=args.family,
        ds_type=args.ds_type,
        valid_ds_type=args.valid_ds_type,
    )
    for k, v in template.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    # Override specific fields from args that should take precedence
    cfg.max_epochs = args.max_epochs
    cfg.bs = args.bs
    cfg.lr = args.lr
    cfg.n_samples = args.n_samples
    cfg.cores = args.cores
    cfg.seed = args.seed

    # Ablation-specific overrides
    cfg.analytical_context = variant
    cfg.r_tag = variant
    cfg.skip_ref = True   # skip pre-training eval — just train + eval at end
    cfg.wandb = False
    cfg.plot = False
    cfg.save_latest = False
    cfg.save_best = False
    cfg.importance = False
    cfg.sir = False

    return cfg


class AblationTrainer(Trainer):
    """Trainer subclass that injects analytical_context into the model config."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        super().__init__(cfg)

    def _initModel(self) -> None:
        from metabeta.utils.config import ApproximatorConfig
        from metabeta.models.approximator import Approximator

        # Load base model config
        model_cfg_path = Path(self.dir, '..', 'configs', 'models', f'{self.cfg.model_id}.yaml')
        base_cfg = modelFromYaml(
            model_cfg_path,
            d_ffx=self.cfg.max_d,
            d_rfx=self.cfg.max_q,
            likelihood_family=self.cfg.likelihood_family,
        )
        # Inject analytical_context
        self.model_cfg = ApproximatorConfig(
            **{**base_cfg.to_dict(), 'analytical_context': self.cfg.analytical_context}
        )
        self.model = Approximator(self.model_cfg).to(self.device)
        if self.cfg.compile and self.device.type == 'cuda':
            self.model.compile()

        import schedulefree

        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.cfg.lr)


def _run_variant(args: argparse.Namespace, variant: str) -> EvaluationSummary:
    print(f'\n{"=" * 60}')
    print(f'  Variant: {variant}')
    print(f'{"=" * 60}')

    setSeed(args.seed)
    cfg = _build_trainer_cfg(args, variant)
    trainer = AblationTrainer(cfg)
    logger.info(trainer.info)

    # Train
    for epoch in range(1, cfg.max_epochs + 1):
        trainer.current_epoch = epoch
        loss_train = trainer.train()
        loss_valid = trainer.valid()
        print(f'  Epoch {epoch:02d}  train={loss_train:.3f}  valid={loss_valid:.3f}')

    # Evaluate
    print(f'\n  Final evaluation ({cfg.n_samples} samples)...')
    eval_summary = trainer.sample()
    print(summaryTable(eval_summary, cfg.likelihood_family))
    trainer.close()
    return eval_summary


def _comparison_table(
    variants: list[str],
    summaries: list[EvaluationSummary],
    likelihood_family: int = 0,
) -> str:
    param_groups = ['ffx', 'sigma_rfx', 'rfx']
    if likelihood_family == 0:
        param_groups.append('sigma_eps')

    # Collect per-group NRMSE and ECE
    headers = ['Variant'] + [f'NRMSE({g})' for g in param_groups] + ['NRMSE(avg)', 'ECE(avg)', 'mLOO-NLL']
    rows = []
    for var, s in zip(variants, summaries):
        row = [var]
        for g in param_groups:
            v = s.nrmse.get(g)
            row.append(f'{torch.mean(v).item():.3f}' if v is not None else '-')
        row.append(f'{dictMean(s.nrmse):.3f}')
        row.append(f'{dictMean(s.ece):.3f}')
        row.append(f'{s.mloonll:.3f}' if s.mloonll is not None else '-')
        rows.append(row)

    return '\n' + tabulate(rows, headers=headers, tablefmt='simple') + '\n'


def main() -> None:
    args = setup()
    setupLogging(args.verbosity)
    torch.set_num_threads(args.cores)

    summaries: list[EvaluationSummary] = []
    completed: list[str] = []

    for variant in args.variants:
        try:
            s = _run_variant(args, variant)
            summaries.append(s)
            completed.append(variant)
        except Exception as e:
            logger.error(f'Variant {variant} failed: {e}')
            raise

    print('\n' + '=' * 70)
    print('  ABLATION COMPARISON')
    print('=' * 70)
    print(_comparison_table(completed, summaries, likelihood_family=args.family))


if __name__ == '__main__':
    main()
