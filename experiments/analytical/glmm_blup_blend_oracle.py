"""Oracle ablation for Priority 3: beta blend for BLUP residuals.

Sweeps alpha_blup over a grid and reports BLUP NRMSE (FFX and sRFX unchanged).
beta_for_blup = alpha_blup * beta_P6 + (1-alpha_blup) * beta_PQL

Run from experiments/analytical/:
    uv run python experiments/analytical/glmm_blup_blend_oracle.py --family b
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
ALPHA_GRID = [0.0, 0.25, 0.5, 0.65, 0.75, 1.0]


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


class _Store:
    def __init__(self) -> None:
        self.blup_errs: list[np.ndarray] = []
        self.blup_truths: list[np.ndarray] = []
        self.ffx_errs: list[np.ndarray] = []
        self.ffx_truths: list[np.ndarray] = []

    def add(
        self, stats: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], max_q: int
    ) -> None:
        mask_d = batch['mask_d'].bool()
        mask_q = batch['mask_q'][..., :max_q].bool()
        mask_m = batch['mask_m'].bool()
        for b in range(batch['X'].shape[0]):
            blup_est = stats['blup_est'][b][mask_m[b]][:, mask_q[b]]
            blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
            self.blup_errs.append((blup_est - blup_true).reshape(-1).cpu().numpy())
            self.blup_truths.append(blup_true.reshape(-1).cpu().numpy())
            self.ffx_errs.append(
                (stats['beta_est'][b][mask_d[b]] - batch['ffx'][b][mask_d[b]]).cpu().numpy()
            )
            self.ffx_truths.append(batch['ffx'][b][mask_d[b]].cpu().numpy())

    def blup_nrmse(self) -> float:
        return _nrmse(np.concatenate(self.blup_errs), np.concatenate(self.blup_truths))

    def ffx_nrmse(self) -> float:
        return _nrmse(np.concatenate(self.ffx_errs), np.concatenate(self.ffx_truths))


def run_oracle(args: argparse.Namespace) -> None:
    family = args.family
    alphas = args.alphas

    header = ','.join(['alpha', 'dataset', 'partition', 'N', 'FFX', 'BLUP'])
    print(header, flush=True)

    for size in args.sizes:
        combos = [(f'{size}-{family}-mixed', 'train', 2)]
        combos.extend((f'{size}-{family}-sampled', part, 0) for part in ['valid', 'test'])

        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            likelihood_family = int(cfg.get('likelihood_family', 0))

            stores = {a: _Store() for a in alphas}

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
                        for alpha in alphas:
                            stats = glmm(
                                batch['X'],
                                batch['y'],
                                Zm,
                                batch['mask_n'].float(),
                                batch['mask_m'].float(),
                                batch['ns'].clamp(min=1).float(),
                                batch['n'].float(),
                                likelihood_family=likelihood_family,
                                map_refine=True,
                                alpha_blup=alpha,
                                **common,
                            )
                            stores[alpha].add(stats, batch, max_q)

            n_total = stores[alphas[0]].blup_errs.__len__()  # approximate
            for alpha in alphas:
                s = stores[alpha]
                n = sum(len(e) for e in s.blup_errs)
                print(
                    f'{alpha:.2f},{data_id},{partition},{n},'
                    f'{s.ffx_nrmse():.4f},{s.blup_nrmse():.4f}',
                    flush=True,
                )


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',      nargs='+', default=SIZES,      choices=SIZES)
    parser.add_argument('--family',                default='b',         choices=['n', 'b'])
    parser.add_argument('--alphas',     nargs='+', type=float, default=ALPHA_GRID)
    parser.add_argument('--batch-size', type=int,  default=32)
    parser.add_argument('--max-batches',type=int,  default=None)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run_oracle(setup())
