"""Diagnostic for analytical BLUP shrinkage errors.

Compares estimated vs true one-dimensional BLUP shrinkage and runs oracle ablations
for the Gaussian analytical estimator. Intended as a follow-up to
metabeta/analytical/plan.md I4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from metabeta.analytical.glmm import glmm
from metabeta.analytical.linalg import _adaptiveRidgeBm, _safeSolve
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import datasetFilename


SIZES = ['small', 'medium', 'large', 'huge']
LAMBDA_RATIO_BINS = [
    (0.0, 0.25, '<0.25'),
    (0.25, 0.5, '0.25-0.5'),
    (0.5, 0.75, '0.5-0.75'),
    (0.75, 1.25, '0.75-1.25'),
    (1.25, 2.0, '1.25-2'),
    (2.0, 4.0, '2-4'),
    (4.0, float('inf'), '>=4'),
]


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    data_dir = ROOT / 'metabeta' / 'outputs' / 'data' / cfg['data_id']
    if partition == 'train':
        return [data_dir / datasetFilename('train', ep) for ep in range(1, n_epochs + 1)]
    return [data_dir / f'{partition}.npz']


def _true_psi(batch: dict[str, torch.Tensor], max_q: int) -> torch.Tensor:
    sigma = batch['sigma_rfx'][..., :max_q]
    corr = batch['corr_rfx'][..., :max_q, :max_q]
    psi = sigma[:, :, None] * corr * sigma[:, None, :]
    if 'mask_q' in batch:
        mask_q = batch['mask_q'][..., :max_q].to(dtype=psi.dtype)
        psi = psi * mask_q[:, :, None] * mask_q[:, None, :]
    return psi


def _blup_for_params(
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    beta: torch.Tensor,
    psi: torch.Tensor,
    sigma_eps: torch.Tensor,
) -> torch.Tensor:
    B, m, _, q = Zm.shape
    active = mask_m.bool()
    eye_q = torch.eye(q, device=Zm.device, dtype=Zm.dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)
    ztz = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ztz_safe = torch.where(active[:, :, None, None], ztz, eye_q)

    vals, vecs = torch.linalg.eigh(psi + sigma_eps.square()[:, None, None] * 1e-4 * eye_q)
    psi_inv = vecs @ torch.diag_embed(1.0 / vals.clamp(min=1e-30)) @ vecs.mT
    inner = sigma_eps.square()[:, None, None, None] * psi_inv[:, None] + ztz_safe
    W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q_bm) * mask_m[:, :, None, None]
    resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta)) * mask_n
    ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    return torch.einsum('bmqr,bmr->bmq', W_g, ztr).nan_to_num().clamp(-20.0, 20.0)


def shrinkage_diagnostic() -> None:
    print('=== q=1 shrinkage diagnostic ===')
    for size in SIZES:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            rows = []
            err_all: list[np.ndarray] = []
            truth_all: list[np.ndarray] = []

            with torch.no_grad():
                for path in _paths(data_id, partition, n_epochs):
                    for batch in Dataloader(path, batch_size=32, shuffle=False):
                        batch = toDevice(batch, torch.device('cpu'))
                        stats = glmm(
                            batch['X'],
                            batch['y'],
                            batch['Z'][..., :max_q],
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            eta_rfx=batch.get('eta_rfx'),
                            mask_q=batch.get('mask_q'),
                        )
                        B = batch['X'].shape[0]
                        q_count = batch['mask_q'][..., :max_q].sum(dim=1)
                        q1 = q_count == 1
                        if not bool(q1.any()):
                            continue

                        z = batch['Z'][q1, ..., :1]
                        mask_n = batch['mask_n'][q1].float()
                        mask_m = batch['mask_m'][q1].bool()
                        z2 = (z.squeeze(-1).square() * mask_n).sum(dim=-1)
                        psi_hat = stats['Psi'][q1, 0, 0].clamp(min=0.0)
                        se2_hat = stats['sigma_eps_est'][q1, 0].square().clamp(min=1e-12)
                        psi_true = batch['sigma_rfx'][q1, 0].square()
                        se2_true = batch['sigma_eps'][q1].square().clamp(min=1e-12)

                        lambda_hat = (
                            psi_hat[:, None]
                            * z2
                            / (se2_hat[:, None] + psi_hat[:, None] * z2).clamp(min=1e-12)
                        )
                        lambda_true = (
                            psi_true[:, None]
                            * z2
                            / (se2_true[:, None] + psi_true[:, None] * z2).clamp(min=1e-12)
                        )
                        lambda_ratio = lambda_hat / lambda_true.clamp(min=1e-8)
                        psi_ratio = psi_hat / psi_true.clamp(min=1e-8)
                        se2_ratio = se2_hat / se2_true

                        blup_err = (stats['blup_est'][q1, :, 0] - batch['rfx'][q1, :, 0]) * mask_m
                        blup_true = batch['rfx'][q1, :, 0] * mask_m
                        beta_err = (
                            (stats['beta_est'] - batch['ffx'])[q1].abs()
                            * batch['mask_d'][q1].float()
                        ).amax(dim=1)

                        for i in range(int(q1.sum())):
                            active_g = mask_m[i].numpy().astype(bool)
                            rows.extend(
                                (
                                    float(lambda_ratio[i, g]),
                                    float(psi_ratio[i]),
                                    float(se2_ratio[i]),
                                    float(beta_err[i]),
                                    float(blup_err[i, g]),
                                    float(blup_true[i, g]),
                                )
                                for g in np.flatnonzero(active_g)
                            )
                            err_all.append(blup_err[i, active_g].numpy())
                            truth_all.append(blup_true[i, active_g].numpy())

            if not rows:
                continue

            arr = np.array(rows)
            global_err = np.concatenate(err_all)
            global_truth = np.concatenate(truth_all)
            scale = max(float(np.std(global_truth)), 1e-8)
            table = []
            for lo, hi, label in LAMBDA_RATIO_BINS:
                sel = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
                if sel.sum() < 20:
                    continue
                table.append(
                    [
                        label,
                        int(sel.sum()),
                        f'{100.0 * sel.mean():.1f}%',
                        f'{np.sqrt(np.mean(arr[sel, 4] ** 2)) / scale:.3f}',
                        f'{np.median(arr[sel, 1]):.3f}',
                        f'{np.median(arr[sel, 2]):.3f}',
                        f'{np.median(arr[sel, 3]):.3f}',
                    ]
                )

            print(
                f'\n{data_id}/{partition} q=1 groups={len(rows)} global={_nrmse(global_err, global_truth):.3f}'
            )
            print(
                tabulate(
                    table,
                    headers=[
                        'lambda_hat/true',
                        'N',
                        'share',
                        'BLUP NRMSE',
                        'med Psi ratio',
                        'med se2 ratio',
                        'med max |beta err|',
                    ],
                    tablefmt='simple',
                )
            )


def oracle_ablation_small_mixed() -> None:
    print('\n=== small-n-mixed oracle BLUP ablations ===')
    data_id = 'small-n-mixed'
    cfg = loadDataConfig(data_id)
    max_q = cfg['max_q']
    cases = {
        'baseline estimator': [],
        'true Psi + true se + beta_hat': [],
        'true Psi + true se + beta_wg': [],
        'true Psi + est se + beta_hat': [],
        'est Psi + true se + beta_hat': [],
        'est Psi + est se + beta_true': [],
        'est Psi + est se + beta_wg': [],
        'true Psi + true se + beta_true': [],
    }
    truth_parts: list[np.ndarray] = []

    with torch.no_grad():
        for path in _paths(data_id, 'train', 2):
            for batch in Dataloader(path, batch_size=32, shuffle=False):
                batch = toDevice(batch, torch.device('cpu'))
                Zm = batch['Z'][..., :max_q]
                mask_n = batch['mask_n'].float()
                mask_m = batch['mask_m'].float()
                stats = glmm(
                    batch['X'],
                    batch['y'],
                    Zm,
                    mask_n,
                    mask_m,
                    batch['ns'].clamp(min=1).float(),
                    batch['n'].float(),
                    eta_rfx=batch.get('eta_rfx'),
                    mask_q=batch.get('mask_q'),
                )
                psi_true = _true_psi(batch, max_q)
                psi_est = stats['Psi']
                se_true = batch['sigma_eps']
                se_est = stats['sigma_eps_est'].squeeze(-1)
                beta_true = batch['ffx']
                beta_hat = stats['beta_est']
                beta_wg = stats['beta_wg']

                blups = {
                    'baseline estimator': stats['blup_est'],
                    'true Psi + true se + beta_hat': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_hat, psi_true, se_true
                    ),
                    'true Psi + true se + beta_wg': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_wg, psi_true, se_true
                    ),
                    'true Psi + est se + beta_hat': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_hat, psi_true, se_est
                    ),
                    'est Psi + true se + beta_hat': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_hat, psi_est, se_true
                    ),
                    'est Psi + est se + beta_true': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_true, psi_est, se_est
                    ),
                    'est Psi + est se + beta_wg': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_wg, psi_est, se_est
                    ),
                    'true Psi + true se + beta_true': _blup_for_params(
                        batch['X'], batch['y'], Zm, mask_n, mask_m, beta_true, psi_true, se_true
                    ),
                }
                active = batch['mask_m'].bool()
                mask_q = batch['mask_q'][..., :max_q].bool()
                for b in range(batch['X'].shape[0]):
                    truth = batch['rfx'][b][active[b]][:, mask_q[b]].reshape(-1).numpy()
                    truth_parts.append(truth)
                    for name, value in blups.items():
                        est = value[b][active[b]][:, mask_q[b]].reshape(-1).numpy()
                        cases[name].append(est - truth)

    truth_all = np.concatenate(truth_parts)
    rows = []
    for name, err_parts in cases.items():
        err = np.concatenate(err_parts)
        rows.append([name, f'{_nrmse(err, truth_all):.4f}', f'{float(np.mean(err)):+.4f}'])
    print(tabulate(rows, headers=['case', 'BLUP NRMSE', 'bias'], tablefmt='simple'))


if __name__ == '__main__':
    shrinkage_diagnostic()
    oracle_ablation_small_mixed()
