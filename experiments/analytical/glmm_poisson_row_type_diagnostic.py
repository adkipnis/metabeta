"""Row-type diagnostics for the retained Poisson GLMM path.

Example:

    uv run python -u experiments/analytical/glmm_poisson_row_type_diagnostic.py \
        --combos small-p-mixed:train:2 medium-p-sampled:valid large-p-sampled:test \
        --max-datasets 1000 --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


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


def _rmse(x: torch.Tensor) -> float:
    return float(x.square().mean().clamp(min=0.0).sqrt().item()) if x.numel() else float('nan')


def _corr(records: list[dict[str, float | int | str]], key: str) -> float:
    x = np.asarray([float(row[key]) for row in records], dtype=float)
    y = np.asarray([float(row['ffx_rmse']) for row in records], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _maskedQuantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return float('nan')
    return float(torch.quantile(selected, q).item())


def _rowRecords(
    data_id: str,
    partition: str,
    offset: int,
    stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
) -> list[dict[str, float | int | str]]:
    Zm = batch['Z'][..., :max_q]
    mask_d = batch['mask_d'].bool()
    mask_q = batch['mask_q'][..., :max_q].bool()
    mask_m = batch['mask_m'].bool()
    mask_n = batch['mask_n'].bool()
    active_q_f = mask_q.to(Zm.dtype)
    sigma2 = stats['sigma_rfx_est'][:, :max_q].square() * active_q_f
    sigma_corr = 0.5 * torch.einsum('bmnq,bq,bmnq->bmn', Zm, sigma2, Zm)
    if 'blup_var' in stats:
        blup_var = stats['blup_var'][..., :max_q] * active_q_f[:, None, :]
        vg_corr = 0.5 * torch.einsum('bmnq,bmq,bmnq->bmn', Zm, blup_var, Zm)
    else:
        vg_corr = sigma_corr.new_zeros(sigma_corr.shape)

    neff = stats.get('poisson_vg_sigma_average_neff')
    adapt = stats.get('poisson_variational_gaussian_adaptive_accept_count')
    beta_step = stats.get('poisson_variational_gaussian_adaptive_beta_step')
    mean_step = stats.get('poisson_variational_gaussian_adaptive_mean_step')
    offset_step = stats.get('poisson_variational_gaussian_adaptive_offset_step')
    sigma_step = stats.get('poisson_variational_gaussian_adaptive_sigma_step')

    rows = []
    for b in range(batch['X'].shape[0]):
        md = mask_d[b]
        mq = mask_q[b]
        mm = mask_m[b]
        rows.append(
            {
                'data_id': data_id,
                'partition': partition,
                'idx': offset + b,
                'n': int(batch['n'][b].item()),
                'm': int(batch['m'][b].item()),
                'd': int(md.sum().item()),
                'q': int(mq.sum().item()),
                'ffx_rmse': _rmse(stats['beta_est'][b][md] - batch['ffx'][b][md]),
                'sigma_rmse': _rmse(stats['sigma_rfx_est'][b][mq] - batch['sigma_rfx'][b][mq]),
                'blup_rmse': _rmse(stats['blup_est'][b][mm][:, mq] - batch['rfx'][b][mm][:, mq]),
                'sigma_corr_q95': _maskedQuantile(sigma_corr[b], mask_n[b], 0.95),
                'vg_corr_q95': _maskedQuantile(vg_corr[b], mask_n[b], 0.95),
                'vg_neff': float(neff[b].item()) if neff is not None else float('nan'),
                'adapt_accept': float(adapt[b].item()) if adapt is not None else float('nan'),
                'adapt_beta_step': float(beta_step[b].item()) if beta_step is not None else 0.0,
                'adapt_mean_step': float(mean_step[b].item()) if mean_step is not None else 0.0,
                'adapt_offset_step': (
                    float(offset_step[b].item()) if offset_step is not None else 0.0
                ),
                'adapt_sigma_step': float(sigma_step[b].item()) if sigma_step is not None else 0.0,
            }
        )
    return rows


def _printGroupSummary(rows: list[dict[str, float | int | str]], keys: tuple[str, ...]) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    print('\nGroup summary by ' + '/'.join(keys), flush=True)
    print('group,N,ffx_rmse_mean,ffx_rmse_q90,sigma_rmse_mean,blup_rmse_mean', flush=True)
    for group, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        ffx = np.asarray([float(row['ffx_rmse']) for row in group_rows], dtype=float)
        sigma = np.asarray([float(row['sigma_rmse']) for row in group_rows], dtype=float)
        blup = np.asarray([float(row['blup_rmse']) for row in group_rows], dtype=float)
        print(
            ','.join(
                [
                    '/'.join(str(x) for x in group),
                    str(len(group_rows)),
                    f'{float(np.nanmean(ffx)):.4f}',
                    f'{float(np.nanquantile(ffx, 0.90)):.4f}',
                    f'{float(np.nanmean(sigma)):.4f}',
                    f'{float(np.nanmean(blup)):.4f}',
                ]
            ),
            flush=True,
        )


def _printDiagnostics(rows: list[dict[str, float | int | str]], top_k: int) -> None:
    print(f'rows={len(rows)}', flush=True)
    for key in [
        'd',
        'q',
        'sigma_rmse',
        'blup_rmse',
        'sigma_corr_q95',
        'vg_corr_q95',
        'vg_neff',
        'adapt_accept',
        'adapt_beta_step',
        'adapt_mean_step',
        'adapt_offset_step',
        'adapt_sigma_step',
    ]:
        print(f'corr(ffx_rmse,{key})={_corr(rows, key):+.3f}', flush=True)

    _printGroupSummary(rows, ('data_id', 'partition'))
    _printGroupSummary(rows, ('d', 'q'))

    print('\nWorst current FFX rows', flush=True)
    print(
        'data,part,idx,d,q,ffx_rmse,sigma_rmse,blup_rmse,sigma_corr_q95,vg_corr_q95,'
        'neff,adapt_accept,adapt_beta,adapt_mean,adapt_offset,adapt_sigma',
        flush=True,
    )
    for row in sorted(rows, key=lambda item: float(item['ffx_rmse']), reverse=True)[:top_k]:
        print(
            ','.join(
                [
                    str(row['data_id']),
                    str(row['partition']),
                    str(row['idx']),
                    str(row['d']),
                    str(row['q']),
                    f'{float(row["ffx_rmse"]):.4f}',
                    f'{float(row["sigma_rmse"]):.4f}',
                    f'{float(row["blup_rmse"]):.4f}',
                    f'{float(row["sigma_corr_q95"]):.3f}',
                    f'{float(row["vg_corr_q95"]):.3f}',
                    f'{float(row["vg_neff"]):.2f}',
                    f'{float(row["adapt_accept"]):.0f}',
                    f'{float(row["adapt_beta_step"]):.3f}',
                    f'{float(row["adapt_mean_step"]):.3f}',
                    f'{float(row["adapt_offset_step"]):.3f}',
                    f'{float(row["adapt_sigma_step"]):.3f}',
                ]
            ),
            flush=True,
        )


def runDiagnostic(args: argparse.Namespace) -> None:
    rows: list[dict[str, float | int | str]] = []
    for data_id, partition, n_epochs in [_parseCombo(combo) for combo in args.combos]:
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
                    stats = glmm(
                        batch['X'],
                        batch['y'],
                        batch['Z'][..., :max_q],
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['ns'].clamp(min=1).float(),
                        batch['n'].float(),
                        likelihood_family=likelihood_family,
                        poisson_laplace_eb_diagnostics=True,
                        **_commonKwargs(batch),
                    )
                    rows.extend(_rowRecords(data_id, partition, n_seen, stats, batch, max_q))
                    n_seen += batch['X'].shape[0]
                    if args.max_datasets is not None and n_seen >= args.max_datasets:
                        break
    _printDiagnostics(rows, args.top_k)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--combos',
        nargs='+',
        default=[
            'small-p-mixed:train:2',
            'small-p-sampled:valid',
            'medium-p-mixed:train:2',
            'medium-p-sampled:valid',
            'large-p-mixed:train:2',
            'large-p-sampled:valid',
        ],
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-datasets', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=20)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    runDiagnostic(setup())
