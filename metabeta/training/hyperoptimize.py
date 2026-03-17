from typing import Callable
import yaml
import argparse
import logging
from pathlib import Path
import copy
import optuna
import wandb

from metabeta.training.train import Trainer
from metabeta.utils.config import (
    modelFromYaml,
    ApproximatorConfig,
    SummarizerConfig,
    PosteriorConfig,
)
from metabeta.utils.evaluation import dictMean
from metabeta.utils.logger import setupLogging


logger = logging.getLogger('hyperoptimize.py')


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hyper', help='load configs/{name}.yaml')
    parser.add_argument('--n_startup_trials', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=100)

    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'configs' / f'{args.name}.yaml'
    with open(path, 'r') as p:
        cfg = yaml.safe_load(p)
    cfg.update(vars(args))
    return argparse.Namespace(**cfg)


class HyperOptimizer:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.dir = Path(__file__).resolve().parent
        self.wandb_run = None

        # data config
        data_cfg_path = Path(self.dir, '..', 'simulation', 'configs', f'{self.cfg.d_tag}.yaml')
        assert data_cfg_path.exists(), f'config file {data_cfg_path} does not exist'
        with open(data_cfg_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            d_ffx, d_rfx = data_cfg['max_d'], data_cfg['max_q']

        # base model config
        model_cfg_path = Path(self.dir, '..', 'models', 'configs', f'{self.cfg.m_tag}.yaml')
        self.model_cfg = modelFromYaml(model_cfg_path, d_ffx=d_ffx, d_rfx=d_rfx).to_dict()

        # optuna study
        sampler = optuna.samplers.TPESampler(
            seed=self.cfg.seed, n_startup_trials=self.cfg.n_startup_trials
        )
        self.study = optuna.create_study(
            study_name=self.cfg.name,
            directions=['minimize', 'minimize'],
            sampler=sampler,
            load_if_exists=True,
        )

        # output dir
        self.out_dir = Path(self.dir, '..', 'outputs', 'optuna', self.cfg.name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # optional wandb study tracking
        if self.cfg.wandb:
            wandb_dir = Path(self.dir, '..', 'outputs', 'wandb')
            wandb_dir.mkdir(parents=True, exist_ok=True)
            self.wandb_run = wandb.init(
                project='metabeta',
                name=self.cfg.name,
                config=vars(self.cfg),
                dir=wandb_dir,
            )

    def suggest(self, trial: optuna.Trial) -> ApproximatorConfig:
        """samples model config"""
        model_cfg = copy.deepcopy(self.model_cfg)
        summarizer = model_cfg.pop('summarizer')
        posterior = model_cfg.pop('posterior')
        subnet = posterior.get('subnet_kwargs', {})

        # summarizer
        summarizer['d_model'] = trial.suggest_categorical('summarizer.d_model', [128, 196, 256])
        d_output = trial.suggest_categorical('summarizer.d_output', [32, 48, 64])
        summarizer['d_output'] = min(d_output, summarizer['d_model'])
        summarizer['d_ff'] = trial.suggest_categorical('summarizer.d_ff', [128, 196, 256])
        summarizer['d_output'] = d_output
        summarizer['n_blocks'] = trial.suggest_int('summarizer.n_blocks', 1, 4)
        n_isab = trial.suggest_int('summarizer.n_isab', 0, 4)
        summarizer['n_isab'] = min(n_isab, summarizer['n_blocks'])
        summarizer['dropout'] = trial.suggest_categorical('summarizer.dropout', [0.0, 0.01, 0.05])
        summarizer = SummarizerConfig(**summarizer)

        # posterior
        posterior['n_blocks'] = trial.suggest_int('posterior.n_blocks', 2, 6)
        subnet['d_ff'] = trial.suggest_categorical('posterior.subnet.d_ff', [128, 196, 256])
        subnet['depth'] = trial.suggest_int('posterior.subnet.depth', 2, 4)
        subnet['dropout'] = trial.suggest_categorical('posterior.subnet.dropout', [0.0, 0.01, 0.05])
        subnet['net_type'] = trial.suggest_categorical('posterior.subnet.type', ['mlp', 'residual'])
        posterior['subnet_kwargs'] = subnet
        posterior = PosteriorConfig(**posterior)
        return ApproximatorConfig(**model_cfg, summarizer=summarizer, posterior=posterior)

    def setObjective(self) -> Callable:
        def objective(trial: optuna.Trial):
            trainer_cfg = copy.deepcopy(self.cfg)
            trainer_cfg.model_cfg = self.suggest(trial)
            trainer_cfg.bs = trial.suggest_categorical('bs', [16, 32, 64, 128])
            trainer_cfg.lr = trial.suggest_float('lr', 1e-4, 5e-3)
            trainer = Trainer(trainer_cfg)
            trainer.go()
            eval_summary = trainer.sample()
            nrmse = dictMean(eval_summary.nrmse)
            alcr = dictMean(eval_summary.abs_lcr)
            trainer.close()
            if self.wandb_run is not None:
                wandb.log(
                    {
                        'trial/nrmse': nrmse,
                        'trial/alcr': alcr,
                        'params': trial.params,
                    },
                    step=trial.number,
                )
            return nrmse, alcr

        return objective

    def saveStudy(self) -> None:
        trials = [
            {
                'number': t.number,
                'values': t.values,
                'params': t.params,
                'state': t.state.name,
            }
            for t in self.study.trials
        ]
        with open(self.out_dir / 'trials.yaml', 'w') as f:
            yaml.safe_dump(trials, f, sort_keys=False)
        with open(self.out_dir / 'pareto.yaml', 'w') as f:
            yaml.safe_dump(
                [
                    {
                        'number': t.number,
                        'values': t.values,
                        'params': t.params,
                    }
                    for t in self.study.best_trials
                ],
                f,
                sort_keys=False,
            )
        print(f'Saved study to {self.out_dir}')

    def go(self) -> None:
        self.study.optimize(self.setObjective(), n_trials=self.cfg.n_trials)
        self.saveStudy()
        if self.wandb_run is not None:
            wandb.finish()


if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    hyper_optimizer = HyperOptimizer(cfg)
    hyper_optimizer.go()
