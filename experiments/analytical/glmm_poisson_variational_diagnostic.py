"""Focused diagnostics for the Poisson variational Gaussian prototype.

Example:

    uv run python -u experiments/analytical/glmm_poisson_variational_diagnostic.py \
        --combos medium-p-sampled:valid large-p-sampled:test --max-datasets 1000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


@dataclass(frozen=True)
class Variant:
    name: str
    kwargs: dict[str, object]


VARIANTS = [
    Variant('current', {}),
    Variant('vg', {'poisson_variational_gaussian': True}),
    Variant(
        'vg_sigma_blend_050',
        {'poisson_variational_gaussian': True, 'poisson_variational_gaussian_sigma_blend': 0.5},
    ),
    Variant(
        'vg_strong_prior',
        {'poisson_variational_gaussian': True, 'poisson_variational_gaussian_prior_weight': 8.0},
    ),
    Variant(
        'vg_no_sigma_avg',
        {'poisson_variational_gaussian': True, 'poisson_laplace_pirls_sigma_average': False},
    ),
]


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')


def _corr(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return float('nan')
    xa = xa[mask]
    ya = ya[mask]
    if float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        return float('nan')
    return float(np.corrcoef(xa, ya)[0, 1])


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _parseCombo(combo: str) -> tuple[str, str, int]:
    parts = combo.split(':')
    if len(parts) not in {2, 3}:
        raise ValueError(f'combo must be data_id:partition[:n_epochs], got {combo!r}')
    return parts[0], parts[1], int(parts[2]) if len(parts) == 3 else 0


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _commonKwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    return {
        'eta_rfx': batch.get('eta_rfx'),
        'mask_q': batch.get('mask_q'),
        'nu_ffx': batch.get('nu_ffx'),
        'tau_ffx': batch.get('tau_ffx'),
        'family_ffx': batch.get('family_ffx'),
        'tau_rfx': batch.get('tau_rfx'),
        'family_sigma_rfx': batch.get('family_sigma_rfx'),
        'tau_eps': batch.get('tau_eps'),
        'family_sigma_eps': batch.get('family_sigma_eps'),
        'mask_d': batch.get('mask_d'),
    }


def _runVariant(
    variant: Variant,
    batch: dict[str, torch.Tensor],
    max_q: int,
    likelihood_family: int,
) -> tuple[dict[str, torch.Tensor], float]:
    kwargs = {
        'poisson_laplace_eb_diagnostics': True,
        **variant.kwargs,
    }
    t0 = time.perf_counter()
    stats = glmm(
        batch['X'],
        batch['y'],
        batch['Z'][..., :max_q],
        batch['mask_n'].float(),
        batch['mask_m'].float(),
        batch['ns'].clamp(min=1).float(),
        batch['n'].float(),
        likelihood_family=likelihood_family,
        **kwargs,
        **_commonKwargs(batch),
    )
    return stats, time.perf_counter() - t0


def _hybridBetaOnly(
    current: dict[str, torch.Tensor],
    vg: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out = dict(current)
    out['beta_est'] = vg['beta_est']
    if 'poisson_variational_gaussian_accept' in vg:
        out['poisson_variational_gaussian_accept'] = vg['poisson_variational_gaussian_accept']
    if 'poisson_variational_gaussian_target' in vg:
        out['poisson_variational_gaussian_target'] = vg['poisson_variational_gaussian_target']
        out['poisson_variational_gaussian_base_target'] = vg[
            'poisson_variational_gaussian_base_target'
        ]
    return out


def _hybridBetaSigma(
    current: dict[str, torch.Tensor],
    vg: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out = dict(current)
    out['beta_est'] = vg['beta_est']
    out['sigma_rfx_est'] = vg['sigma_rfx_est']
    if 'Psi_lap' in vg:
        out['Psi_lap'] = vg['Psi_lap']
    if 'poisson_variational_gaussian_accept' in vg:
        out['poisson_variational_gaussian_accept'] = vg['poisson_variational_gaussian_accept']
    return out


def _appendMetrics(
    store: dict[str, list[np.ndarray | float]],
    stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
    elapsed: float,
) -> None:
    B = batch['X'].shape[0]
    mask_d = batch['mask_d'].bool()
    mask_q = batch['mask_q'][..., :max_q].bool()
    mask_m = batch['mask_m'].bool()
    store.setdefault('wall', []).extend([elapsed / max(B, 1)] * B)
    for b in range(B):
        store.setdefault('beta_err', []).append(
            (stats['beta_est'][b][mask_d[b]] - batch['ffx'][b][mask_d[b]]).cpu().numpy()
        )
        store.setdefault('beta_true', []).append(batch['ffx'][b][mask_d[b]].cpu().numpy())
        store.setdefault('sigma_err', []).append(
            (stats['sigma_rfx_est'][b][mask_q[b]] - batch['sigma_rfx'][b][mask_q[b]]).cpu().numpy()
        )
        store.setdefault('sigma_true', []).append(batch['sigma_rfx'][b][mask_q[b]].cpu().numpy())
        blup_est = stats['blup_est'][b][mask_m[b]][:, mask_q[b]]
        blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
        store.setdefault('blup_err', []).append((blup_est - blup_true).reshape(-1).cpu().numpy())
        store.setdefault('blup_true', []).append(blup_true.reshape(-1).cpu().numpy())


def _marginalCorrectionSummary(
    stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
) -> tuple[np.ndarray, np.ndarray]:
    Z = batch['Z'][..., :max_q]
    mask_n = batch['mask_n'].bool()
    mask_q = batch['mask_q'][..., :max_q].bool()
    sigma2 = stats['sigma_rfx_est'][:, :max_q].square() * mask_q.to(Z.dtype)
    sigma_corr = 0.5 * torch.einsum('bmnq,bq,bmnq->bmn', Z, sigma2, Z)
    if 'blup_var' in stats:
        blup_var = stats['blup_var'][..., :max_q] * mask_q[:, None, :].to(Z.dtype)
        var_corr = 0.5 * torch.einsum('bmnq,bmq,bmnq->bmn', Z, blup_var, Z)
    else:
        var_corr = sigma_corr.new_zeros(sigma_corr.shape)

    sigma_q95 = []
    var_q95 = []
    for b in range(Z.shape[0]):
        active = mask_n[b]
        sigma_q95.append(float(torch.quantile(sigma_corr[b][active], 0.95).cpu()))
        var_q95.append(float(torch.quantile(var_corr[b][active], 0.95).cpu()))
    return np.asarray(sigma_q95), np.asarray(var_q95)


def _rowRecords(
    data_id: str,
    partition: str,
    offset: int,
    current: dict[str, torch.Tensor],
    vg: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
) -> list[dict[str, float | int | str]]:
    records = []
    mask_d = batch['mask_d'].bool()
    mask_q = batch['mask_q'][..., :max_q].bool()
    mask_m = batch['mask_m'].bool()
    cur_sigma_q95, cur_var_q95 = _marginalCorrectionSummary(current, batch, max_q)
    vg_sigma_q95, vg_var_q95 = _marginalCorrectionSummary(vg, batch, max_q)
    accept = vg.get('poisson_variational_gaussian_accept')
    target = vg.get('poisson_variational_gaussian_target')
    base_target = vg.get('poisson_variational_gaussian_base_target')
    neff = current.get('poisson_sigma_average_neff')
    best_scale = current.get('poisson_sigma_average_best_scale')
    for b in range(batch['X'].shape[0]):
        md = mask_d[b]
        mq = mask_q[b]
        mm = mask_m[b]
        beta_true = batch['ffx'][b][md]
        sigma_true = batch['sigma_rfx'][b][mq]
        blup_true = batch['rfx'][b][mm][:, mq]
        cur_beta_rmse = _rmse((current['beta_est'][b][md] - beta_true).cpu().numpy())
        vg_beta_rmse = _rmse((vg['beta_est'][b][md] - beta_true).cpu().numpy())
        cur_sigma_rmse = _rmse((current['sigma_rfx_est'][b][mq] - sigma_true).cpu().numpy())
        vg_sigma_rmse = _rmse((vg['sigma_rfx_est'][b][mq] - sigma_true).cpu().numpy())
        cur_blup_rmse = _rmse((current['blup_est'][b][mm][:, mq] - blup_true).cpu().numpy())
        vg_blup_rmse = _rmse((vg['blup_est'][b][mm][:, mq] - blup_true).cpu().numpy())
        records.append(
            {
                'data_id': data_id,
                'partition': partition,
                'idx': offset + b,
                'n': int(batch['n'][b].item()),
                'm': int(batch['m'][b].item()),
                'd': int(md.sum().item()),
                'q': int(mq.sum().item()),
                'delta_ffx': vg_beta_rmse - cur_beta_rmse,
                'delta_sigma': vg_sigma_rmse - cur_sigma_rmse,
                'delta_blup': vg_blup_rmse - cur_blup_rmse,
                'cur_ffx': cur_beta_rmse,
                'vg_ffx': vg_beta_rmse,
                'cur_sigma': cur_sigma_rmse,
                'vg_sigma': vg_sigma_rmse,
                'accept': float(accept[b].item()) if accept is not None else float('nan'),
                'target_delta': (
                    float((target[b] - base_target[b]).item())
                    if target is not None and base_target is not None
                    else float('nan')
                ),
                'neff': float(neff[b].item()) if neff is not None else float('nan'),
                'best_scale': float(best_scale[b].item())
                if best_scale is not None
                else float('nan'),
                'cur_sigma_q95': float(cur_sigma_q95[b]),
                'vg_sigma_q95': float(vg_sigma_q95[b]),
                'cur_var_q95': float(cur_var_q95[b]),
                'vg_var_q95': float(vg_var_q95[b]),
            }
        )
    return records


def _printVariantSummary(name: str, store: dict[str, list[np.ndarray | float]]) -> None:
    beta_err = np.concatenate(store['beta_err'])
    beta_true = np.concatenate(store['beta_true'])
    sigma_err = np.concatenate(store['sigma_err'])
    sigma_true = np.concatenate(store['sigma_true'])
    blup_err = np.concatenate(store['blup_err'])
    blup_true = np.concatenate(store['blup_true'])
    wall = np.asarray(store['wall'], dtype=float)
    print(
        ','.join(
            [
                name,
                f'{_nrmse(beta_err, beta_true):.4f}',
                f'{_nrmse(sigma_err, sigma_true):.4f}',
                f'{_nrmse(blup_err, blup_true):.4f}',
                f'{1000.0 * float(np.mean(wall)):.2f}',
            ]
        ),
        flush=True,
    )


def _printDeltaSummary(records: list[dict[str, float | int | str]]) -> None:
    delta_ffx = [float(r['delta_ffx']) for r in records]
    delta_sigma = [float(r['delta_sigma']) for r in records]
    delta_blup = [float(r['delta_blup']) for r in records]
    accept = [float(r['accept']) for r in records]
    target_delta = [float(r['target_delta']) for r in records]
    neff = [float(r['neff']) for r in records]
    sigma_q95 = [float(r['cur_sigma_q95']) for r in records]
    var_q95_delta = [float(r['vg_var_q95']) - float(r['cur_var_q95']) for r in records]
    n = len(records)
    ffx_better = np.asarray(delta_ffx) < 0.0
    sigma_worse = np.asarray(delta_sigma) > 0.0
    blup_worse = np.asarray(delta_blup) > 0.0
    print('\nVG row-level delta diagnostics', flush=True)
    print(f'rows={n}', flush=True)
    print(f'ffx_better={float(ffx_better.mean()):.3f}', flush=True)
    print(f'ffx_better_sigma_worse={float((ffx_better & sigma_worse).mean()):.3f}', flush=True)
    print(f'ffx_better_blup_worse={float((ffx_better & blup_worse).mean()):.3f}', flush=True)
    print(f'accept_mean={float(np.nanmean(accept)):.3f}', flush=True)
    print(f'target_delta_mean={float(np.nanmean(target_delta)):.3f}', flush=True)
    print(f'delta_ffx_mean={float(np.nanmean(delta_ffx)):+.4f}', flush=True)
    print(f'delta_sigma_mean={float(np.nanmean(delta_sigma)):+.4f}', flush=True)
    print(f'delta_blup_mean={float(np.nanmean(delta_blup)):+.4f}', flush=True)
    print(f'corr(delta_ffx,delta_sigma)={_corr(delta_ffx, delta_sigma):+.3f}', flush=True)
    print(f'corr(delta_ffx,target_delta)={_corr(delta_ffx, target_delta):+.3f}', flush=True)
    print(f'corr(delta_ffx,neff)={_corr(delta_ffx, neff):+.3f}', flush=True)
    print(f'corr(delta_ffx,current_sigma_q95)={_corr(delta_ffx, sigma_q95):+.3f}', flush=True)
    print(f'corr(delta_ffx,delta_var_q95)={_corr(delta_ffx, var_q95_delta):+.3f}', flush=True)

    def print_top(title: str, key: str, reverse: bool = True) -> None:
        selected = sorted(records, key=lambda r: float(r[key]), reverse=reverse)[:8]
        print(f'\n{title}', flush=True)
        print(
            'data,part,idx,d,q,delta_ffx,delta_sigma,delta_blup,accept,target_delta,neff,'
            'best_scale,sigma_q95,var_q95',
            flush=True,
        )
        for r in selected:
            print(
                ','.join(
                    [
                        str(r['data_id']),
                        str(r['partition']),
                        str(r['idx']),
                        str(r['d']),
                        str(r['q']),
                        f'{float(r["delta_ffx"]):+.4f}',
                        f'{float(r["delta_sigma"]):+.4f}',
                        f'{float(r["delta_blup"]):+.4f}',
                        f'{float(r["accept"]):.0f}',
                        f'{float(r["target_delta"]):+.2f}',
                        f'{float(r["neff"]):.2f}',
                        f'{float(r["best_scale"]):.2f}',
                        f'{float(r["cur_sigma_q95"]):.3f}',
                        f'{float(r["cur_var_q95"]):.3f}',
                    ]
                ),
                flush=True,
            )

    print_top('Largest VG FFX regressions', 'delta_ffx', reverse=True)
    joint = [r for r in records if float(r['delta_ffx']) < 0.0 and float(r['delta_sigma']) > 0.0]
    if joint:
        print_top('Largest sigma regressions among FFX gains', 'delta_sigma', reverse=True)


def runDiagnostic(args: argparse.Namespace) -> None:
    combos = [_parseCombo(combo) for combo in args.combos]
    stores: dict[str, dict[str, list[np.ndarray | float]]] = {
        variant.name: {} for variant in VARIANTS
    }
    stores['vg_beta_only'] = {}
    stores['vg_beta_sigma'] = {}
    row_records: list[dict[str, float | int | str]] = []

    for data_id, partition, n_epochs in combos:
        cfg = loadDataConfig(data_id)
        max_q = cfg['max_q']
        likelihood_family = int(cfg.get('likelihood_family', 0))
        n_seen = 0
        with torch.no_grad():
            for path in _paths(data_id, partition, n_epochs):
                for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                    batch = toDevice(batch, torch.device('cpu'))
                    if args.max_datasets is not None:
                        remaining = args.max_datasets - n_seen
                        if remaining <= 0:
                            break
                        if batch['X'].shape[0] > remaining:
                            batch = _sliceBatch(batch, remaining)

                    stats_by_name = {}
                    elapsed_by_name = {}
                    for variant in VARIANTS:
                        stats, elapsed = _runVariant(variant, batch, max_q, likelihood_family)
                        stats_by_name[variant.name] = stats
                        elapsed_by_name[variant.name] = elapsed
                        _appendMetrics(stores[variant.name], stats, batch, max_q, elapsed)

                    beta_only = _hybridBetaOnly(stats_by_name['current'], stats_by_name['vg'])
                    beta_sigma = _hybridBetaSigma(stats_by_name['current'], stats_by_name['vg'])
                    _appendMetrics(
                        stores['vg_beta_only'],
                        beta_only,
                        batch,
                        max_q,
                        elapsed_by_name['vg'],
                    )
                    _appendMetrics(
                        stores['vg_beta_sigma'],
                        beta_sigma,
                        batch,
                        max_q,
                        elapsed_by_name['vg'],
                    )
                    row_records.extend(
                        _rowRecords(
                            data_id,
                            partition,
                            n_seen,
                            stats_by_name['current'],
                            stats_by_name['vg'],
                            batch,
                            max_q,
                        )
                    )
                    n_seen += batch['X'].shape[0]
                    if args.max_datasets is not None and n_seen >= args.max_datasets:
                        break

    print('variant,FFX,sRFX,BLUP,ms_per_ds', flush=True)
    for name in [
        'current',
        'vg',
        'vg_beta_only',
        'vg_beta_sigma',
        'vg_sigma_blend_050',
        'vg_strong_prior',
        'vg_no_sigma_avg',
    ]:
        _printVariantSummary(name, stores[name])
    _printDeltaSummary(row_records)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--combos',
        nargs='+',
        default=['medium-p-sampled:valid', 'large-p-sampled:test'],
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-datasets', type=int, default=1000)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    runDiagnostic(setup())
