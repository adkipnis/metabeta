"""Compact benchmark for the required GLMM error-analysis suite.

Supports normal (family=n) and Bernoulli (family=b) likelihood families.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from metabeta.analytical.glmm import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


SIZES = ['small', 'medium', 'large', 'huge']
METHODS = ['default', 'current', 'raw', 'bernoulli_eb', 'normal_eb']
FAMILIES = ['n', 'b']


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def run_required_benchmark(args: argparse.Namespace) -> None:
    family = args.family
    print(
        'method,dataset,partition,N,FFX,sRFX,sEps,BLUP,ms_per_ds,gate,accept,'
        'blup_fallback,beta_capped,sigma_capped',
        flush=True,
    )
    combos = _combos(args, family)
    for data_id, partition, n_epochs in combos:
        cfg = loadDataConfig(data_id)
        max_q = cfg['max_q']
        likelihood_family = int(cfg.get('likelihood_family', 0))
        stores = {method: _MetricStore(likelihood_family) for method in args.methods}

        with torch.no_grad():
            n_seen = 0
            for path in _paths(data_id, partition, n_epochs):
                for batch_idx, batch in enumerate(
                    Dataloader(path, batch_size=args.batch_size, shuffle=False)
                ):
                    if args.max_batches is not None and batch_idx >= args.max_batches:
                        break
                    batch = toDevice(batch, torch.device('cpu'))
                    if args.max_datasets is not None:
                        remaining = args.max_datasets - n_seen
                        if remaining <= 0:
                            break
                        if batch['X'].shape[0] > remaining:
                            batch = _sliceBatch(batch, remaining)
                    Zm = batch['Z'][..., :max_q]
                    common = dict(
                        eta_rfx=batch.get('eta_rfx'),
                        mask_q=batch.get('mask_q'),
                        nu_ffx=batch.get('nu_ffx'),
                        tau_ffx=batch.get('tau_ffx'),
                        family_ffx=batch.get('family_ffx'),
                        tau_rfx=batch.get('tau_rfx'),
                        family_sigma_rfx=batch.get('family_sigma_rfx'),
                        tau_eps=batch.get('tau_eps'),
                        family_sigma_eps=batch.get('family_sigma_eps'),
                        mask_d=batch.get('mask_d'),
                    )
                    for method in args.methods:
                        t0 = time.perf_counter()
                        stats = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            likelihood_family=likelihood_family,
                            map_refine=method != 'raw',
                            **_methodKwargs(method),
                            bernoulli_laplace_eb_diagnostics=(
                                method in {'default', 'current', 'bernoulli_eb'}
                            ),
                            **_bernoulliEbKwargs(method, args),
                            **_normalEbKwargs(method, args),
                            **common,
                        )
                        stores[method].add(stats, batch, max_q, time.perf_counter() - t0)
                    n_seen += batch['X'].shape[0]

        for method in args.methods:
            print(stores[method].row(method, data_id, partition), flush=True)


def _combos(args: argparse.Namespace, family: str) -> list[tuple[str, str, int]]:
    if args.combos:
        return [_parseCombo(combo) for combo in args.combos]
    out = []
    for size in args.sizes:
        out.append((f'{size}-{family}-mixed', 'train', 2))
        out.extend((f'{size}-{family}-sampled', part, 0) for part in ['valid', 'test'])
    return out


def _parseCombo(combo: str) -> tuple[str, str, int]:
    parts = combo.split(':')
    if len(parts) not in {2, 3}:
        raise ValueError(f'combo must be data_id:partition[:n_epochs], got {combo!r}')
    n_epochs = int(parts[2]) if len(parts) == 3 else 0
    return parts[0], parts[1], n_epochs


class _MetricStore:
    def __init__(self, likelihood_family: int = 0) -> None:
        self.likelihood_family = likelihood_family
        self.beta_errs: list[np.ndarray] = []
        self.beta_truths: list[np.ndarray] = []
        self.srfx_errs: list[np.ndarray] = []
        self.srfx_truths: list[np.ndarray] = []
        self.seps_errs: list[np.ndarray] = []
        self.seps_truths: list[np.ndarray] = []
        self.blup_errs: list[np.ndarray] = []
        self.blup_truths: list[np.ndarray] = []
        self.wall: list[float] = []
        self.gate: list[np.ndarray] = []
        self.accept: list[np.ndarray] = []
        self.blup_fallback: list[np.ndarray] = []
        self.beta_capped: list[np.ndarray] = []
        self.sigma_capped: list[np.ndarray] = []
        self.n_total = 0

    def add(
        self,
        stats: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        max_q: int,
        elapsed: float,
    ) -> None:
        B = batch['X'].shape[0]
        mask_d = batch['mask_d'].bool()
        mask_q = batch['mask_q'][..., :max_q].bool()
        mask_m = batch['mask_m'].bool()
        self.wall.extend([elapsed / max(B, 1)] * B)
        if 'laplace_eb_gate' in stats:
            self.gate.append(stats['laplace_eb_gate'].detach().cpu().numpy())
        if 'laplace_eb_accept' in stats:
            self.accept.append(stats['laplace_eb_accept'].detach().cpu().numpy())
        if 'normal_laplace_eb_accept' in stats:
            self.accept.append(stats['normal_laplace_eb_accept'].detach().cpu().numpy())
        if 'laplace_eb_blup_fallback' in stats:
            self.blup_fallback.append(stats['laplace_eb_blup_fallback'].detach().cpu().numpy())
        if 'laplace_eb_beta_output_capped' in stats:
            self.beta_capped.append(stats['laplace_eb_beta_output_capped'].detach().cpu().numpy())
        if 'normal_map_beta_prior_capped' in stats:
            self.beta_capped.append(stats['normal_map_beta_prior_capped'].detach().cpu().numpy())
        if 'laplace_eb_sigma_prior_capped' in stats:
            self.sigma_capped.append(stats['laplace_eb_sigma_prior_capped'].detach().cpu().numpy())
        self.n_total += B
        for b in range(B):
            self.beta_errs.append(
                (stats['beta_est'][b][mask_d[b]] - batch['ffx'][b][mask_d[b]]).cpu().numpy()
            )
            self.beta_truths.append(batch['ffx'][b][mask_d[b]].cpu().numpy())
            self.srfx_errs.append(
                (stats['sigma_rfx_est'][b][mask_q[b]] - batch['sigma_rfx'][b][mask_q[b]])
                .cpu()
                .numpy()
            )
            self.srfx_truths.append(batch['sigma_rfx'][b][mask_q[b]].cpu().numpy())
            if self.likelihood_family == 0:
                self.seps_errs.append(
                    (stats['sigma_eps_est'][b, 0] - batch['sigma_eps'][b]).reshape(1).cpu().numpy()
                )
                self.seps_truths.append(batch['sigma_eps'][b].reshape(1).cpu().numpy())
            blup_est = stats['blup_est'][b][mask_m[b]][:, mask_q[b]]
            blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
            self.blup_errs.append((blup_est - blup_true).reshape(-1).cpu().numpy())
            self.blup_truths.append(blup_true.reshape(-1).cpu().numpy())

    def row(self, method: str, data_id: str, partition: str) -> str:
        gate = float(np.mean(np.concatenate(self.gate))) if self.gate else float('nan')
        accept = float(np.mean(np.concatenate(self.accept))) if self.accept else float('nan')
        blup_fallback = (
            float(np.mean(np.concatenate(self.blup_fallback)))
            if self.blup_fallback
            else float('nan')
        )
        beta_capped = (
            float(np.mean(np.concatenate(self.beta_capped))) if self.beta_capped else float('nan')
        )
        sigma_capped = (
            float(np.mean(np.concatenate(self.sigma_capped))) if self.sigma_capped else float('nan')
        )
        seps_str = (
            f'{_nrmse(np.concatenate(self.seps_errs), np.concatenate(self.seps_truths)):.4f}'
            if self.likelihood_family == 0
            else 'n/a'
        )
        return ','.join(
            [
                method,
                data_id,
                partition,
                str(self.n_total),
                f'{_nrmse(np.concatenate(self.beta_errs), np.concatenate(self.beta_truths)):.4f}',
                f'{_nrmse(np.concatenate(self.srfx_errs), np.concatenate(self.srfx_truths)):.4f}',
                seps_str,
                f'{_nrmse(np.concatenate(self.blup_errs), np.concatenate(self.blup_truths)):.4f}',
                f'{1000.0 * float(np.mean(self.wall)):.2f}',
                f'{gate:.3f}',
                f'{accept:.3f}',
                f'{blup_fallback:.3f}',
                f'{beta_capped:.3f}',
                f'{sigma_capped:.3f}',
            ]
        )


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _methodKwargs(method: str) -> dict[str, str | bool]:
    if method in {'default', 'current'}:
        return {}
    if method == 'raw':
        return {'bernoulli_laplace_eb': False, 'normal_laplace_eb': False}
    if method == 'bernoulli_eb':
        return {'bernoulli_laplace_eb': 'bernoulli_eb', 'normal_laplace_eb': False}
    if method == 'normal_eb':
        return {'bernoulli_laplace_eb': False, 'normal_laplace_eb': True}
    return {'bernoulli_laplace_eb': False}


def _bernoulliEbKwargs(method: str, args: argparse.Namespace) -> dict[str, int | float | bool]:
    if method == 'bernoulli_eb':
        return {
            'bernoulli_laplace_eb_sigma_prior_cap': args.bernoulli_eb_sigma_prior_cap,
            'bernoulli_laplace_eb_sigma_prior_cap_min_d': (args.bernoulli_eb_sigma_prior_cap_min_d),
        }
    if method in {'default', 'current'}:
        return {
            'bernoulli_laplace_eb_sigma_prior_cap': args.bernoulli_eb_sigma_prior_cap,
            'bernoulli_laplace_eb_sigma_prior_cap_min_d': (args.bernoulli_eb_sigma_prior_cap_min_d),
        }
    return {}


def _normalEbKwargs(method: str, args: argparse.Namespace) -> dict[str, int | float | str]:
    if method not in {'default', 'current', 'normal_eb'}:
        return {}
    return {
        'normal_laplace_eb_mode': args.normal_eb_mode,
        'normal_laplace_eb_steps': args.normal_eb_steps,
        'normal_laplace_eb_lr': args.normal_eb_lr,
        'normal_laplace_eb_moment_blend': args.normal_eb_moment_blend,
        'normal_laplace_eb_prior_weight': args.normal_eb_prior_weight,
    }


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',      nargs='+', default=SIZES,      choices=SIZES)
    parser.add_argument('--methods',    nargs='+', default=['current'], choices=METHODS)
    parser.add_argument('--family',                default='n',         choices=FAMILIES)
    parser.add_argument('--batch-size', type=int,  default=32)
    parser.add_argument('--max-batches',type=int,  default=None)
    parser.add_argument('--max-datasets', type=int, default=None)
    parser.add_argument('--combos',     nargs='+', default=None)
    parser.add_argument('--bernoulli-eb-sigma-prior-cap', type=float, default=2.5)
    parser.add_argument('--bernoulli-eb-sigma-prior-cap-min-d', type=int, default=5)
    parser.add_argument('--normal-eb-mode', default='moment', choices=['moment', 'gradient'])
    parser.add_argument('--normal-eb-steps', type=int, default=3)
    parser.add_argument('--normal-eb-lr', type=float, default=0.08)
    parser.add_argument('--normal-eb-moment-blend', type=float, default=1.0)
    parser.add_argument('--normal-eb-prior-weight', type=float, default=4.0)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run_required_benchmark(setup())
