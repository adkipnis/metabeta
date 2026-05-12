"""MAP optimizer ablation diagnostic for the Gaussian analytical GLMM.

Compares four MAP optimization variants to check whether jointly optimizing
beta and/or sigma(Eps) alongside sigma(RFX) improves quality or just adds
cost and variance.

Methods:
  raw         — raw MoM/EM, no MAP (reference baseline)
  map_rfx     — MAP optimizes sigma(RFX) only; beta and sigma(Eps) fixed
  map_rfx_eps — MAP optimizes sigma(RFX) + sigma(Eps); beta fixed
  map_rfx_beta — MAP optimizes sigma(RFX) + beta; sigma(Eps) fixed
  current     — production MAP: all three optimized jointly
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from metabeta.analytical.glmm import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


SIZES = ['small', 'medium', 'large', 'huge']
METHODS = ['raw', 'map_rfx', 'map_rfx_eps', 'map_rfx_beta', 'current']

# Maps method name -> (map_refine, map_optimize)
_METHOD_KWARGS: dict[str, dict] = {
    'raw': {'map_refine': False},
    'map_rfx': {'map_refine': True, 'map_optimize': 'rfx'},
    'map_rfx_eps': {'map_refine': True, 'map_optimize': 'rfx_eps'},
    'map_rfx_beta': {'map_refine': True, 'map_optimize': 'rfx_beta'},
    'current': {'map_refine': True, 'map_optimize': 'all'},
}


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


class _MetricStore:
    def __init__(self) -> None:
        self.beta_errs: list[np.ndarray] = []
        self.beta_truths: list[np.ndarray] = []
        self.srfx_errs: list[np.ndarray] = []
        self.srfx_truths: list[np.ndarray] = []
        self.seps_errs: list[np.ndarray] = []
        self.seps_truths: list[np.ndarray] = []
        self.blup_errs: list[np.ndarray] = []
        self.blup_truths: list[np.ndarray] = []
        self.n_total = 0

    def add(
        self, stats: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], max_q: int
    ) -> None:
        mask_d = batch['mask_d'].bool()
        mask_q = batch['mask_q'][..., :max_q].bool()
        mask_m = batch['mask_m'].bool()
        self.n_total += batch['X'].shape[0]
        for b in range(batch['X'].shape[0]):
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
            self.seps_errs.append(
                (stats['sigma_eps_est'][b, 0] - batch['sigma_eps'][b]).reshape(1).cpu().numpy()
            )
            self.seps_truths.append(batch['sigma_eps'][b].reshape(1).cpu().numpy())
            blup_est = stats['blup_est'][b][mask_m[b]][:, mask_q[b]]
            blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
            self.blup_errs.append((blup_est - blup_true).reshape(-1).cpu().numpy())
            self.blup_truths.append(blup_true.reshape(-1).cpu().numpy())

    def nrmse(self) -> tuple[float, float, float, float]:
        ffx = _nrmse(np.concatenate(self.beta_errs), np.concatenate(self.beta_truths))
        srfx = _nrmse(np.concatenate(self.srfx_errs), np.concatenate(self.srfx_truths))
        seps = _nrmse(np.concatenate(self.seps_errs), np.concatenate(self.seps_truths))
        blup = _nrmse(np.concatenate(self.blup_errs), np.concatenate(self.blup_truths))
        return ffx, srfx, seps, blup

    def row(self, method: str, data_id: str, partition: str) -> str:
        ffx, srfx, seps, blup = self.nrmse()
        return f'{method},{data_id},{partition},{self.n_total},{ffx:.4f},{srfx:.4f},{seps:.4f},{blup:.4f}'


def run_ablation(args: argparse.Namespace) -> None:
    all_rows: dict[str, list[tuple[str, str, float, float, float, float]]] = {
        m: [] for m in args.methods
    }

    print('method,dataset,partition,N,FFX,sRFX,sEps,BLUP', flush=True)
    for size in args.sizes:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            stores = {method: _MetricStore() for method in args.methods}

            with torch.no_grad():
                for path in _paths(data_id, partition, n_epochs):
                    for batch_idx, batch in enumerate(
                        Dataloader(path, batch_size=args.batch_size, shuffle=False)
                    ):
                        if args.max_batches is not None and batch_idx >= args.max_batches:
                            break
                        batch = toDevice(batch, torch.device('cpu'))
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
                            stats = glmm(
                                batch['X'],
                                batch['y'],
                                Zm,
                                batch['mask_n'].float(),
                                batch['mask_m'].float(),
                                batch['ns'].clamp(min=1).float(),
                                batch['n'].float(),
                                **_METHOD_KWARGS[method],
                                **common,
                            )
                            stores[method].add(stats, batch, max_q)

            for method in args.methods:
                row = stores[method].row(method, data_id, partition)
                print(row, flush=True)
                ffx, srfx, seps, blup = stores[method].nrmse()
                all_rows[method].append((data_id, partition, ffx, srfx, seps, blup))

    _print_totals(all_rows, args.methods)


def _print_totals(
    all_rows: dict[str, list[tuple[str, str, float, float, float, float]]],
    methods: list[str],
) -> None:
    print('\n--- weighted row totals (NRMSE, row-mean) ---')
    print(f'{"method":<16}  {"FFX":>7}  {"sRFX":>7}  {"sEps":>7}  {"BLUP":>7}')
    for method in methods:
        rows = all_rows[method]
        if not rows:
            continue
        ffx = float(np.mean([r[2] for r in rows]))
        srfx = float(np.mean([r[3] for r in rows]))
        seps = float(np.mean([r[4] for r in rows]))
        blup = float(np.mean([r[5] for r in rows]))
        print(f'{method:<16}  {ffx:7.4f}  {srfx:7.4f}  {seps:7.4f}  {blup:7.4f}')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=SIZES, choices=SIZES)
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-batches', type=int, default=None)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run_ablation(setup())
