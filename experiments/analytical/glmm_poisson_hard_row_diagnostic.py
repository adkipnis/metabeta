"""Hard-row Poisson diagnostic for the remaining INLA FFX gap.

The script ranks rows by the row-level FFX gap between the current best Poisson
prototype and INLA, then reruns the selected rows through oracle and higher-budget
variants:

- true-σ fixed VG refresh;
- INLA-σ fixed VG refresh, when INLA estimates are available;
- wider VG-centered σ averaging;
- VG budget ladder and target-calibration variants.

Example:

    uv run python -u experiments/analytical/glmm_poisson_hard_row_diagnostic.py \
        --combos medium-p-sampled:valid large-p-sampled:test \
        --max-datasets 1000 --top-k 64
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from metabeta.analytical.fit import glmm
from metabeta.analytical.glmm.poisson import refinePoissonVariationalGaussianSigmaAverage
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


@dataclass(frozen=True)
class RowKey:
    data_id: str
    partition: str
    idx: int


@dataclass(frozen=True)
class RankedRow:
    key: RowKey
    d: int
    q: int
    current_rmse: float
    vg_rmse: float
    inla_rmse: float
    gap_to_inla: float


@dataclass(frozen=True)
class Variant:
    name: str
    kwargs: dict[str, object]


VARIANTS = [
    Variant('current', {}),
    Variant(
        'vg_sigma_avg',
        {'poisson_variational_gaussian': True, 'poisson_variational_gaussian_sigma_average': True},
    ),
    Variant(
        'vg_wide_sigma_avg',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_sigma_average': True,
            'poisson_variational_gaussian_sigma_average_scales': (
                0.25,
                0.5,
                0.75,
                1.0,
                1.3333333,
                2.0,
                3.0,
                4.0,
            ),
            'poisson_variational_gaussian_sigma_average_steps': 3,
            'poisson_variational_gaussian_sigma_average_temperature': 4.0,
        },
    ),
    Variant(
        'vg_sigma_avg_beta_sigma',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_sigma_average': True,
            'poisson_variational_gaussian_sigma_average_output_mode': 'beta_sigma',
        },
    ),
    Variant(
        'vg_state_avg',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_sigma_average': True,
            'poisson_variational_gaussian_state_average': True,
        },
    ),
    Variant(
        'vg_state_avg_beta_sigma',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_sigma_average': True,
            'poisson_variational_gaussian_state_average': True,
            'poisson_variational_gaussian_state_average_output_mode': 'beta_sigma',
        },
    ),
    Variant(
        'vg_outer7',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_outer': 7,
            'poisson_variational_gaussian_inner': 3,
            'poisson_variational_gaussian_final': 2,
        },
    ),
    Variant(
        'vg_inner3_legacy',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_outer': 5,
            'poisson_variational_gaussian_inner': 3,
            'poisson_variational_gaussian_final': 2,
        },
    ),
    Variant(
        'vg_final5',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_outer': 5,
            'poisson_variational_gaussian_inner': 3,
            'poisson_variational_gaussian_final': 5,
        },
    ),
    Variant(
        'vg_sigma_blend_010',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_sigma_blend': 0.1,
        },
    ),
    Variant(
        'vg_sigma_blend_050',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_sigma_blend': 0.5,
        },
    ),
    Variant(
        'vg_prior_2',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_prior_weight': 2.0,
        },
    ),
    Variant(
        'vg_prior_8',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_prior_weight': 8.0,
        },
    ),
    Variant(
        'vg_high_budget',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_outer': 10,
            'poisson_variational_gaussian_inner': 4,
            'poisson_variational_gaussian_final': 2,
        },
    ),
    Variant(
        'vg_high_budget_sigma_avg',
        {
            'poisson_variational_gaussian': True,
            'poisson_variational_gaussian_outer': 10,
            'poisson_variational_gaussian_inner': 4,
            'poisson_variational_gaussian_final': 2,
            'poisson_variational_gaussian_sigma_average': True,
            'poisson_variational_gaussian_sigma_average_scales': (
                0.25,
                0.5,
                0.75,
                1.0,
                1.3333333,
                2.0,
                3.0,
                4.0,
            ),
            'poisson_variational_gaussian_sigma_average_steps': 3,
            'poisson_variational_gaussian_sigma_average_temperature': 4.0,
        },
    ),
]


def _parseCombo(combo: str) -> tuple[str, str, int]:
    parts = combo.split(':')
    if len(parts) not in {2, 3}:
        raise ValueError(f'combo must be data_id:partition[:n_epochs], got {combo!r}')
    return parts[0], parts[1], int(parts[2]) if len(parts) == 3 else 0


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _inlaPath(path: Path) -> Path:
    return path.with_suffix('.inla.npz')


def _loadInla(path: Path) -> dict[str, np.ndarray] | None:
    inla_path = _inlaPath(path)
    if not inla_path.exists():
        return None
    with np.load(inla_path) as f:
        return {
            'beta': f['inla_ffx'].copy(),
            'sigma_rfx': f['inla_sigma_rfx'].copy(),
            'blups': f['inla_rfx'].copy(),
            'failed': f['inla_failed'].copy().astype(bool),
        }


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _selectBatch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value.index_select(0, indices)
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


def _runGlmm(
    batch: dict[str, torch.Tensor],
    max_q: int,
    likelihood_family: int,
    kwargs: dict[str, object],
) -> tuple[dict[str, torch.Tensor], float]:
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
        poisson_laplace_eb_diagnostics=True,
        **kwargs,
        **_commonKwargs(batch),
    )
    return stats, time.perf_counter() - t0


def _rmse(x: torch.Tensor | np.ndarray) -> float:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float('nan')


def _rowBetaRmse(
    beta_est: torch.Tensor,
    beta_true: torch.Tensor,
    active_d: torch.Tensor,
) -> float:
    return _rmse(beta_est[active_d] - beta_true[active_d])


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
    if 'poisson_vg_sigma_average_neff' in stats:
        store.setdefault('neff', []).append(
            stats['poisson_vg_sigma_average_neff'].detach().cpu().numpy()
        )
    if 'poisson_vg_sigma_average_best_scale' in stats:
        store.setdefault('best_scale', []).append(
            stats['poisson_vg_sigma_average_best_scale'].detach().cpu().numpy()
        )
    for b in range(B):
        md = mask_d[b]
        mq = mask_q[b]
        mm = mask_m[b]
        store.setdefault('beta_err', []).append(
            (stats['beta_est'][b][md] - batch['ffx'][b][md]).detach().cpu().numpy()
        )
        store.setdefault('beta_true', []).append(batch['ffx'][b][md].detach().cpu().numpy())
        store.setdefault('sigma_err', []).append(
            (stats['sigma_rfx_est'][b][mq] - batch['sigma_rfx'][b][mq]).detach().cpu().numpy()
        )
        store.setdefault('sigma_true', []).append(batch['sigma_rfx'][b][mq].detach().cpu().numpy())
        blup_est = stats['blup_est'][b][: mm.shape[0]][mm][:, mq]
        blup_true = batch['rfx'][b][mm][:, mq]
        store.setdefault('blup_err', []).append((blup_est - blup_true).reshape(-1).cpu().numpy())
        store.setdefault('blup_true', []).append(blup_true.reshape(-1).detach().cpu().numpy())


def _nrmse(errs: list[np.ndarray | float], truths: list[np.ndarray | float]) -> float:
    err = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in errs])
    truth = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in truths])
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _summarize(name: str, store: dict[str, list[np.ndarray | float]]) -> str:
    wall = np.asarray(store.get('wall', []), dtype=float)
    neff_values = store.get('neff', [])
    best_scale_values = store.get('best_scale', [])
    neff = float(np.nanmean(np.concatenate(neff_values))) if neff_values else float('nan')
    best_scale = (
        float(np.nanmean(np.concatenate(best_scale_values))) if best_scale_values else float('nan')
    )
    return ','.join(
        [
            name,
            f'{_nrmse(store["beta_err"], store["beta_true"]):.4f}',
            f'{_nrmse(store["sigma_err"], store["sigma_true"]):.4f}',
            f'{_nrmse(store["blup_err"], store["blup_true"]):.4f}',
            f'{1000.0 * float(np.mean(wall)):.2f}' if wall.size else 'nan',
            f'{neff:.2f}',
            f'{best_scale:.2f}',
        ]
    )


def _priorKwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    return {
        'nu_ffx': batch.get('nu_ffx'),
        'tau_ffx': batch.get('tau_ffx'),
        'family_ffx': batch.get('family_ffx'),
        'tau_rfx': batch.get('tau_rfx'),
        'family_sigma_rfx': batch.get('family_sigma_rfx'),
        'mask_d': batch.get('mask_d'),
        'mask_q': batch.get('mask_q'),
    }


def _sigmaPlugInRefresh(
    stats: dict[str, torch.Tensor],
    sigma: torch.Tensor,
    batch: dict[str, torch.Tensor],
    max_q: int,
    n_steps: int,
) -> dict[str, torch.Tensor]:
    out = dict(stats)
    q = max_q
    sigma_out = out['sigma_rfx_est'].clone()
    sigma_out[:, :q] = sigma[:, :q].to(device=sigma_out.device, dtype=sigma_out.dtype)
    out['sigma_rfx_est'] = sigma_out
    out['Psi_lap'] = torch.diag_embed(sigma_out[:, :q].square())
    return refinePoissonVariationalGaussianSigmaAverage(
        out,
        batch['X'],
        batch['y'],
        batch['Z'][..., :max_q],
        batch['mask_n'].float(),
        batch['mask_m'].float(),
        **_priorKwargs(batch),
        scales=(1.0,),
        n_steps=n_steps,
        damping=0.5,
        temperature=1.0,
        output_mode='beta',
        return_diagnostics=True,
    )


def _rankValue(row: RankedRow, rank_by: str) -> float:
    if rank_by == 'vg_rmse':
        return row.vg_rmse
    if rank_by == 'current_rmse':
        return row.current_rmse
    if rank_by == 'gap_to_inla':
        return row.gap_to_inla
    raise ValueError("rank_by must be 'vg_rmse', 'current_rmse', or 'gap_to_inla'")


def _rankRows(args: argparse.Namespace) -> list[RankedRow]:
    ranked = []
    with torch.no_grad():
        for combo in args.combos:
            data_id, partition, n_epochs = _parseCombo(combo)
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            likelihood_family = int(cfg.get('likelihood_family', 0))
            n_seen = 0
            for path in _paths(data_id, partition, n_epochs):
                inla = _loadInla(path)
                for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                    batch = toDevice(batch, torch.device('cpu'))
                    if args.max_datasets is not None:
                        remaining = args.max_datasets - n_seen
                        if remaining <= 0:
                            break
                        if batch['X'].shape[0] > remaining:
                            batch = _sliceBatch(batch, remaining)

                    current, _ = _runGlmm(batch, max_q, likelihood_family, {})
                    vg, _ = _runGlmm(
                        batch,
                        max_q,
                        likelihood_family,
                        {
                            'poisson_variational_gaussian': True,
                            'poisson_variational_gaussian_sigma_average': True,
                        },
                    )
                    mask_d = batch['mask_d'].bool()
                    mask_q = batch['mask_q'][..., :max_q].bool()
                    B = batch['X'].shape[0]
                    for b in range(B):
                        file_idx = n_seen + b
                        if inla is None or bool(inla['failed'][file_idx]):
                            continue
                        md = mask_d[b]
                        inla_beta = torch.as_tensor(
                            inla['beta'][file_idx, md.cpu().numpy()],
                            device=batch['X'].device,
                            dtype=batch['X'].dtype,
                        )
                        true_beta = batch['ffx'][b][md]
                        inla_rmse = _rmse(inla_beta - true_beta)
                        vg_rmse = _rowBetaRmse(vg['beta_est'][b], batch['ffx'][b], md)
                        ranked.append(
                            RankedRow(
                                key=RowKey(data_id, partition, file_idx),
                                d=int(md.sum().item()),
                                q=int(mask_q[b].sum().item()),
                                current_rmse=_rowBetaRmse(
                                    current['beta_est'][b], batch['ffx'][b], md
                                ),
                                vg_rmse=vg_rmse,
                                inla_rmse=inla_rmse,
                                gap_to_inla=vg_rmse - inla_rmse,
                            )
                        )
                    n_seen += B
                    if args.max_datasets is not None and n_seen >= args.max_datasets:
                        break
    return sorted(ranked, key=lambda row: _rankValue(row, args.rank_by), reverse=True)[: args.top_k]


def _evaluateSelected(args: argparse.Namespace, selected: list[RankedRow]) -> None:
    selected_map = {(row.key.data_id, row.key.partition, row.key.idx) for row in selected}
    stores = {variant.name: {} for variant in VARIANTS}
    stores['true_sigma_refresh'] = {}
    stores['inla_sigma_refresh'] = {}
    stores['inla'] = {}

    with torch.no_grad():
        for combo in args.combos:
            data_id, partition, n_epochs = _parseCombo(combo)
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            likelihood_family = int(cfg.get('likelihood_family', 0))
            n_seen = 0
            for path in _paths(data_id, partition, n_epochs):
                inla = _loadInla(path)
                for batch in Dataloader(path, batch_size=args.batch_size, shuffle=False):
                    batch = toDevice(batch, torch.device('cpu'))
                    B0 = batch['X'].shape[0]
                    idxs = [
                        b for b in range(B0) if (data_id, partition, n_seen + b) in selected_map
                    ]
                    if not idxs:
                        n_seen += B0
                        continue
                    idx_tensor = torch.as_tensor(idxs, dtype=torch.long)
                    row_indices = np.asarray([n_seen + b for b in idxs], dtype=int)
                    sub = _selectBatch(batch, idx_tensor)

                    stats_by_name = {}
                    elapsed_by_name = {}
                    for variant in VARIANTS:
                        stats, elapsed = _runGlmm(sub, max_q, likelihood_family, variant.kwargs)
                        stats_by_name[variant.name] = stats
                        elapsed_by_name[variant.name] = elapsed
                        _appendMetrics(stores[variant.name], stats, sub, max_q, elapsed)

                    true_refresh = _sigmaPlugInRefresh(
                        stats_by_name['vg_sigma_avg'],
                        sub['sigma_rfx'][..., :max_q],
                        sub,
                        max_q,
                        args.oracle_steps,
                    )
                    _appendMetrics(
                        stores['true_sigma_refresh'],
                        true_refresh,
                        sub,
                        max_q,
                        elapsed_by_name['vg_sigma_avg'],
                    )

                    if inla is not None:
                        inla_sigma = torch.as_tensor(
                            inla['sigma_rfx'][row_indices, :max_q],
                            device=sub['X'].device,
                            dtype=sub['X'].dtype,
                        )
                        inla_refresh = _sigmaPlugInRefresh(
                            stats_by_name['vg_sigma_avg'],
                            inla_sigma,
                            sub,
                            max_q,
                            args.oracle_steps,
                        )
                        _appendMetrics(
                            stores['inla_sigma_refresh'],
                            inla_refresh,
                            sub,
                            max_q,
                            elapsed_by_name['vg_sigma_avg'],
                        )
                        inla_stats = {
                            'beta_est': torch.as_tensor(
                                inla['beta'][row_indices],
                                device=sub['X'].device,
                                dtype=sub['X'].dtype,
                            ),
                            'sigma_rfx_est': torch.as_tensor(
                                inla['sigma_rfx'][row_indices],
                                device=sub['X'].device,
                                dtype=sub['X'].dtype,
                            ),
                            'blup_est': torch.as_tensor(
                                inla['blups'][row_indices],
                                device=sub['X'].device,
                                dtype=sub['X'].dtype,
                            ),
                        }
                        _appendMetrics(stores['inla'], inla_stats, sub, max_q, 0.0)
                    n_seen += B0

    positive_gap = np.asarray([row.gap_to_inla for row in selected], dtype=float) > 0.0
    print('\nSelected hard rows', flush=True)
    print(
        f'rank_by={args.rank_by}, rows={len(selected)}, '
        f'positive_vg_gap_to_inla={float(positive_gap.mean()):.3f}',
        flush=True,
    )
    print('data,part,idx,d,q,current_rmse,vg_rmse,inla_rmse,vg_gap_to_inla', flush=True)
    for row in selected[:20]:
        print(
            ','.join(
                [
                    row.key.data_id,
                    row.key.partition,
                    str(row.key.idx),
                    str(row.d),
                    str(row.q),
                    f'{row.current_rmse:.4f}',
                    f'{row.vg_rmse:.4f}',
                    f'{row.inla_rmse:.4f}',
                    f'{row.gap_to_inla:+.4f}',
                ]
            ),
            flush=True,
        )

    print('\nvariant,FFX,sRFX,BLUP,ms_per_ds,neff,best_scale', flush=True)
    for name in [
        'current',
        'vg_sigma_avg',
        'vg_wide_sigma_avg',
        'vg_sigma_avg_beta_sigma',
        'vg_state_avg',
        'vg_state_avg_beta_sigma',
        'vg_outer7',
        'vg_inner3_legacy',
        'vg_final5',
        'vg_sigma_blend_010',
        'vg_sigma_blend_050',
        'vg_prior_2',
        'vg_prior_8',
        'vg_high_budget',
        'vg_high_budget_sigma_avg',
        'true_sigma_refresh',
        'inla_sigma_refresh',
        'inla',
    ]:
        if stores[name]:
            print(_summarize(name, stores[name]), flush=True)


def runDiagnostic(args: argparse.Namespace) -> None:
    selected = _rankRows(args)
    if not selected:
        print('No selected rows. Check that INLA files exist for the requested combos.', flush=True)
        return
    _evaluateSelected(args, selected)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--combos',
        nargs='+',
        default=['medium-p-sampled:valid', 'large-p-sampled:test'],
    )
    parser.add_argument('--max-datasets', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--oracle-steps', type=int, default=4)
    parser.add_argument(
        '--rank-by',
        choices=['vg_rmse', 'current_rmse', 'gap_to_inla'],
        default='vg_rmse',
    )
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    runDiagnostic(setup())
