"""Diagnose where fast normal GLMM estimates lag diagonal R-INLA.

This is intentionally small-sample: R-INLA is seconds per dataset, so the script
is for failure-mode attribution, not the required benchmark table.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tabulate import tabulate

from experiments.analytical.glmm_inla_comparison import _flatten, _inla_estimate
from metabeta.analytical.glmm import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


DATA_IDS = ['small-n-mixed', 'medium-n-mixed', 'large-n-mixed', 'huge-n-mixed']
METHODS = ['normal_eb']


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')


def _binByD(d: int) -> str:
    if d <= 4:
        return 'd<=4'
    if d <= 8:
        return '5<=d<=8'
    if d <= 12:
        return '9<=d<=12'
    return 'd>=13'


def _binByM(m: int) -> str:
    if m <= 20:
        return 'm<=20'
    if m <= 60:
        return '21<=m<=60'
    return 'm>=61'


def _methodKwargs(method: str) -> dict[str, bool]:
    if method == 'normal_eb':
        return {'map_refine': True, 'bernoulli_laplace_eb': False, 'normal_laplace_eb': True}
    raise ValueError(f'unsupported analytical method: {method}')


def runDiagnostic(args: argparse.Namespace) -> None:
    rows = []
    t0 = time.perf_counter()
    for data_id in args.data_ids:
        cfg = loadDataConfig(data_id)
        max_q = cfg['max_q']
        paths = (
            [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, args.n_epochs + 1)]
            if args.partition == 'train'
            else [dataFilePath(cfg['data_id'], args.partition)]
        )
        seen = 0
        for path in paths:
            if seen >= args.n_inla:
                break
            for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                if seen >= args.n_inla:
                    break
                if seen + batch['X'].shape[0] > args.n_inla:
                    batch = _sliceBatch(batch, args.n_inla - seen)
                batch = toDevice(batch, torch.device('cpu'))
                B = batch['X'].shape[0]
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
                stats = {
                    method: glmm(
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['ns'].clamp(min=1).float(),
                        batch['n'].float(),
                        likelihood_family=0,
                        **_methodKwargs(method),
                        **common,
                    )
                    for method in METHODS
                }
                ffx_true = batch['ffx'].cpu().numpy()
                srfx_true = batch['sigma_rfx'].cpu().numpy()
                rfx_true = batch['rfx'].cpu().numpy()
                mask_d = batch['mask_d'].cpu().numpy().astype(bool)
                mask_q = batch['mask_q'].cpu().numpy().astype(bool)
                m_arr = batch['m'].cpu().numpy()

                for b in range(B):
                    active_d = np.flatnonzero(mask_d[b])
                    active_q = np.flatnonzero(mask_q[b])
                    m = int(m_arr[b])
                    est = _inla_estimate(
                        _flatten(batch, b, active_d, active_q),
                        likelihood_family=0,
                        normal_re_correlation='diagonal',
                    )
                    seen += 1
                    if est is None:
                        continue
                    inla_err = {
                        'ffx': _rmse(est['beta'] - ffx_true[b, active_d]),
                        'sigma': _rmse(est['sigma_rfx'] - srfx_true[b, active_q]),
                        'blup': _rmse(
                            (est['blups'][:m] - rfx_true[b, :m][:, active_q]).reshape(-1)
                        ),
                    }
                    for method in METHODS:
                        method_err = {
                            'ffx': _rmse(
                                stats[method]['beta_est'][b, active_d].detach().cpu().numpy()
                                - ffx_true[b, active_d]
                            ),
                            'sigma': _rmse(
                                stats[method]['sigma_rfx_est'][b, active_q].detach().cpu().numpy()
                                - srfx_true[b, active_q]
                            ),
                            'blup': _rmse(
                                (
                                    stats[method]['blup_est'][b, :m][:, active_q]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    - rfx_true[b, :m][:, active_q]
                                ).reshape(-1)
                            ),
                        }
                        for metric, value in method_err.items():
                            rows.append(
                                {
                                    'data_id': data_id,
                                    'method': method,
                                    'metric': metric,
                                    'method_rmse': value,
                                    'inla_rmse': inla_err[metric],
                                    'd_bin': _binByD(len(active_d)),
                                    'm_bin': _binByM(m),
                                    'q': len(active_q),
                                }
                            )

    print(f'completed {len(rows)} method/metric rows in {time.perf_counter() - t0:.1f}s')
    _printSummary(rows, 'data_id')
    _printSummary(rows, 'd_bin')
    _printSummary(rows, 'm_bin')


def _printSummary(rows: list[dict], by: str) -> None:
    table = []
    keys = sorted({row[by] for row in rows})
    for key in keys:
        for method in METHODS:
            for metric in ['ffx', 'sigma', 'blup']:
                selected = [
                    row
                    for row in rows
                    if row[by] == key and row['method'] == method and row['metric'] == metric
                ]
                if not selected:
                    continue
                method_rmse = np.array([row['method_rmse'] for row in selected])
                inla_rmse = np.array([row['inla_rmse'] for row in selected])
                delta = method_rmse - inla_rmse
                table.append(
                    [
                        key,
                        method,
                        metric,
                        len(selected),
                        f'{float(method_rmse.mean()):.4f}',
                        f'{float(inla_rmse.mean()):.4f}',
                        f'{float(delta.mean()):+.4f}',
                        f'{float(np.mean(delta > 0.0)):.2f}',
                    ]
                )
    print(f'\nGrouped by {by}')
    print(
        tabulate(
            table,
            headers=['bin', 'method', 'metric', 'N', 'method RMSE', 'INLA RMSE', 'Δ', 'worse%'],
            tablefmt='github',
        )
    )


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-ids',   nargs='+', default=DATA_IDS)
    parser.add_argument('--partition',            default='train')
    parser.add_argument('--n-epochs',   type=int,  default=2)
    parser.add_argument('--n-inla',     type=int,  default=10)
    parser.add_argument('--batch-size', type=int,  default=16)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    runDiagnostic(setup())
