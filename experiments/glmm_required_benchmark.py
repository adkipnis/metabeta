"""Compact benchmark for the required Gaussian GLMM error-analysis suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.glmm import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import datasetFilename


SIZES = ['small', 'medium', 'large', 'huge']


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    data_dir = ROOT / 'metabeta' / 'outputs' / 'data' / cfg['data_id']
    if partition == 'train':
        return [data_dir / datasetFilename('train', ep) for ep in range(1, n_epochs + 1)]
    return [data_dir / f'{partition}.npz']


def run_required_benchmark() -> None:
    print('dataset,partition,N,FFX,sRFX,sEps,BLUP')
    for size in SIZES:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            beta_errs: list[np.ndarray] = []
            beta_truths: list[np.ndarray] = []
            srfx_errs: list[np.ndarray] = []
            srfx_truths: list[np.ndarray] = []
            seps_errs: list[np.ndarray] = []
            seps_truths: list[np.ndarray] = []
            blup_errs: list[np.ndarray] = []
            blup_truths: list[np.ndarray] = []
            n_total = 0

            with torch.no_grad():
                for path in _paths(data_id, partition, n_epochs):
                    for batch in Dataloader(path, batch_size=32, shuffle=False):
                        batch = toDevice(batch, torch.device('cpu'))
                        Zm = batch['Z'][..., :max_q]
                        stats = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            eta_rfx=batch.get('eta_rfx'),
                            mask_q=batch.get('mask_q'),
                        )
                        mask_d = batch['mask_d'].bool()
                        mask_q = batch['mask_q'][..., :max_q].bool()
                        mask_m = batch['mask_m'].bool()
                        n_total += batch['X'].shape[0]
                        for b in range(batch['X'].shape[0]):
                            beta_errs.append(
                                (stats['beta_est'][b][mask_d[b]] - batch['ffx'][b][mask_d[b]])
                                .cpu()
                                .numpy()
                            )
                            beta_truths.append(batch['ffx'][b][mask_d[b]].cpu().numpy())
                            srfx_errs.append(
                                (
                                    stats['sigma_rfx_est'][b][mask_q[b]]
                                    - batch['sigma_rfx'][b][mask_q[b]]
                                )
                                .cpu()
                                .numpy()
                            )
                            srfx_truths.append(batch['sigma_rfx'][b][mask_q[b]].cpu().numpy())
                            seps_errs.append(
                                (stats['sigma_eps_est'][b, 0] - batch['sigma_eps'][b])
                                .reshape(1)
                                .cpu()
                                .numpy()
                            )
                            seps_truths.append(batch['sigma_eps'][b].reshape(1).cpu().numpy())
                            blup_est = stats['blup_est'][b][mask_m[b]][:, mask_q[b]]
                            blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
                            blup_errs.append((blup_est - blup_true).reshape(-1).cpu().numpy())
                            blup_truths.append(blup_true.reshape(-1).cpu().numpy())

            print(
                ','.join(
                    [
                        data_id,
                        partition,
                        str(n_total),
                        f'{_nrmse(np.concatenate(beta_errs), np.concatenate(beta_truths)):.4f}',
                        f'{_nrmse(np.concatenate(srfx_errs), np.concatenate(srfx_truths)):.4f}',
                        f'{_nrmse(np.concatenate(seps_errs), np.concatenate(seps_truths)):.4f}',
                        f'{_nrmse(np.concatenate(blup_errs), np.concatenate(blup_truths)):.4f}',
                    ]
                )
            )


if __name__ == '__main__':
    run_required_benchmark()
