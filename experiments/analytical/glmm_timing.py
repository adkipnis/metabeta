"""Timing experiment: per-stage GLMM cost inside vs outside gradient context.

Hypothesis: the training slowdown from map_refine=True is dominated by
per-batch overhead from N_steps × backward() calls and fresh Adam state
allocation, not by any gradient leakage into NPE parameters.  This overhead
is present regardless of whether an outer gradient context is active (since
inner torch.enable_grad() overrides outer no_grad), but its magnitude should
scale linearly with map_steps.

Methodology
-----------
For each (variant, context) pair, time K batches and report median ms/dataset.
On CUDA, torch.cuda.synchronize() is called around each timed call so that
async kernel dispatch time is not measured instead of actual execution time.

Variants
  raw          lmmNormal only          (map_refine=False)
  map_5        +refineNormalMapSrfx    (map_steps=5,  no EB)
  map_10       +refineNormalMapSrfx    (map_steps=10, no EB)
  map_20       +refineNormalMapSrfx    (map_steps=20, no EB)   ← default steps
  eb_only      +refineNormalLaplaceEb  only (map_steps=0 via 1-step map skipped)
  default      map_steps=20 + EB                               ← full default

Contexts
  no_grad      torch.no_grad()    simulates inference / pre-computation
  grad         grad enabled       simulates the NPE training loop

Usage
-----
  uv run python experiments/analytical/glmm_timing.py
  uv run python experiments/analytical/glmm_timing.py --device cuda --data-id medium-n-sampled
  uv run python experiments/analytical/glmm_timing.py --data-id medium-n-sampled --batches 30

When --device cuda, a CPU vs CUDA comparison section is appended automatically.
The CPU batches are the same data re-pinned to CPU, so the comparison is apples-to-apples.
A ratio > 1.0 means CUDA is slower (kernel launch overhead dominates at this problem size).

A third section measures the transfer overhead of the approximator.py fix: GLMM inputs
moved to CPU, outputs moved back to CUDA.  This is the actual cost paid during training.
"""

from __future__ import annotations

import argparse
import time

import torch
from tabulate import tabulate

from metabeta.analytical.glmm import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: list[tuple[str, dict]] = [
    ('raw',     dict(map_refine=False)),
    ('map_5',   dict(map_refine=True,  normal_laplace_eb=False, map_steps=5)),
    ('map_10',  dict(map_refine=True,  normal_laplace_eb=False, map_steps=10)),
    ('map_20',  dict(map_refine=True,  normal_laplace_eb=False, map_steps=20)),
    ('eb_only', dict(map_refine=True,  normal_laplace_eb=True,  map_steps=1,
                     normal_laplace_eb_mode='moment')),
    ('default', dict(map_refine=True,  normal_laplace_eb=True,  map_steps=20,
                     normal_laplace_eb_mode='moment')),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_common(batch: dict[str, torch.Tensor], max_q: int) -> dict:
    return dict(
        nu_ffx=batch['nu_ffx'],
        tau_ffx=batch['tau_ffx'],
        family_ffx=batch['family_ffx'],
        tau_rfx=batch['tau_rfx'],
        family_sigma_rfx=batch['family_sigma_rfx'],
        tau_eps=batch['tau_eps'],
        family_sigma_eps=batch['family_sigma_eps'],
        mask_d=batch.get('mask_d'),
        eta_rfx=batch.get('eta_rfx'),
        mask_q=batch.get('mask_q'),
    )


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _batch_to(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _run_batch(
    batch: dict[str, torch.Tensor],
    max_q: int,
    likelihood_family: int,
    variant_kwargs: dict,
    device: torch.device,
    glmm_on_cpu: bool = False,
) -> float:
    """Run glmm on one batch, return wall seconds.

    CUDA operations are asynchronous: without synchronize() the clock would
    only capture kernel dispatch time, not actual execution time.

    glmm_on_cpu: move inputs to CPU, run glmm, move stats back to device —
    mirrors the approximator.py fix so the transfer overhead is included.
    """
    Zm = batch['Z'][..., :max_q]
    common = _build_common(batch, max_q)

    if glmm_on_cpu:
        cpu = torch.device('cpu')
        Zm = Zm.cpu()
        common = _batch_to(common, cpu)
        X  = batch['X'].cpu()
        y  = batch['y'].cpu()
        mn = batch['mask_n'].cpu().float()
        mm = batch['mask_m'].cpu().float()
        ns = batch['ns'].cpu().clamp(min=1).float()
        n  = batch['n'].cpu().float()
    else:
        X  = batch['X']
        y  = batch['y']
        mn = batch['mask_n'].float()
        mm = batch['mask_m'].float()
        ns = batch['ns'].clamp(min=1).float()
        n  = batch['n'].float()

    _sync(device)
    t0 = time.perf_counter()
    stats = glmm(X, y, Zm, mn, mm, ns, n,
                 likelihood_family=likelihood_family, **common, **variant_kwargs)
    if glmm_on_cpu:
        # move outputs back — this is part of the cost paid during training
        _ = {k: v.to(device) if torch.is_tensor(v) else v for k, v in stats.items()}
    _sync(device)
    return time.perf_counter() - t0


def _time_variant(
    batches: list[dict[str, torch.Tensor]],
    max_q: int,
    likelihood_family: int,
    variant_kwargs: dict,
    device: torch.device,
    warmup: int,
    glmm_on_cpu: bool = False,
) -> tuple[float, float]:
    """Return (median_ms_per_ds, std_ms_per_ds) over batches."""
    times_ms: list[float] = []

    for i, batch in enumerate(batches):
        with torch.no_grad():
            elapsed = _run_batch(
                batch, max_q, likelihood_family, variant_kwargs, device, glmm_on_cpu
            )
        if i < warmup:
            continue
        n_ds = batch['X'].shape[0]
        times_ms.append(1000.0 * elapsed / max(n_ds, 1))

    arr = torch.tensor(times_ms)
    return float(arr.median()), float(arr.std())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    # CUDA needs more warmup batches: first calls JIT-compile kernels.
    warmup = 5 if device.type == 'cuda' else 2
    cfg = loadDataConfig(args.data_id)
    max_q = cfg['max_q']
    likelihood_family = int(cfg.get('likelihood_family', 0))
    path = dataFilePath(cfg['data_id'], args.partition)

    dl = Dataloader(path, batch_size=args.batch_size, shuffle=False)
    batches: list[dict[str, torch.Tensor]] = []
    for i, batch in enumerate(dl):
        if i >= args.batches + warmup:
            break
        batches.append(toDevice(batch, device))

    n_actual = max(0, len(batches) - warmup)
    device_label = args.device
    if device.type == 'cuda':
        device_label = f'cuda ({torch.cuda.get_device_name(device)})'
    print(f'\nDevice  : {device_label}')
    print(f'Dataset : {args.data_id}  partition={args.partition}')
    print(f'Batches : {n_actual} timed  (+{warmup} warmup)  batch_size={args.batch_size}')
    print(f'Likelihood family: {likelihood_family}\n')

    # --- section 1: on-device timing
    meds: dict[str, float] = {}
    rows = []
    for v_name, v_kwargs in VARIANTS:
        med, std = _time_variant(
            batches, max_q, likelihood_family, v_kwargs, device, warmup,
        )
        meds[v_name] = med
        rows.append([v_name, f'{med:.2f}', f'{std:.2f}'])

    print(tabulate(rows, headers=['variant', 'med ms/ds', 'std ms/ds'], tablefmt='simple'))

    raw_t = meds['raw']
    print('\nIncremental cost per MAP step:')
    inc_rows = [
        ['raw → map_5  (+5)',  f'{meds["map_5"]  - raw_t:.2f}', f'{(meds["map_5"]  - raw_t)/5:.3f}'],
        ['map_5 → map_10 (+5)', f'{meds["map_10"] - meds["map_5"]:.2f}',
         f'{(meds["map_10"] - meds["map_5"])/5:.3f}'],
        ['map_10 → map_20 (+10)', f'{meds["map_20"] - meds["map_10"]:.2f}',
         f'{(meds["map_20"] - meds["map_10"])/10:.3f}'],
    ]
    print(tabulate(inc_rows, headers=['interval', 'delta ms/ds', 'ms/ds per step'],
                   tablefmt='simple'))
    print(f'\ndefault vs raw: {meds["default"] / raw_t:.1f}x')

    if device.type != 'cuda':
        return

    # --- section 2: CPU vs CUDA (no transfer overhead)
    print('\n' + '=' * 60)
    print('CPU vs CUDA  (pure compute, no transfer)')
    print('CUDA/CPU ratio > 1.0 means CUDA is slower')
    print('=' * 60 + '\n')

    cpu_device = torch.device('cpu')
    cpu_batches = [_batch_to(b, cpu_device) for b in batches]
    cpu_meds: dict[str, float] = {}
    cmp_rows = []
    for v_name, v_kwargs in VARIANTS:
        cpu_med, cpu_std = _time_variant(
            cpu_batches, max_q, likelihood_family, v_kwargs,
            cpu_device, warmup=2,
        )
        cpu_meds[v_name] = cpu_med
        ratio = meds[v_name] / cpu_med if cpu_med > 0 else float('nan')
        cmp_rows.append([v_name, f'{cpu_med:.2f}', f'{meds[v_name]:.2f}', f'{ratio:.2f}'])

    print(tabulate(cmp_rows,
                   headers=['variant', 'CPU ms/ds', 'CUDA ms/ds', 'CUDA/CPU'],
                   tablefmt='simple'))

    # --- section 3: CUDA with inputs pinned to CPU (mirrors the approximator fix)
    print('\n' + '=' * 60)
    print('CUDA training step cost: GLMM on CPU + transfer outputs back to GPU')
    print('(mirrors the map_refine fix in approximator.py)')
    print('ratio = (CPU compute + transfer) / pure CUDA')
    print('=' * 60 + '\n')

    fix_rows = []
    for v_name, v_kwargs in VARIANTS:
        fix_med, fix_std = _time_variant(
            batches, max_q, likelihood_family, v_kwargs,
            device, warmup, glmm_on_cpu=True,
        )
        ratio_vs_cuda = fix_med / meds[v_name] if meds[v_name] > 0 else float('nan')
        ratio_vs_cpu  = fix_med / cpu_meds[v_name] if cpu_meds[v_name] > 0 else float('nan')
        fix_rows.append([
            v_name,
            f'{cpu_meds[v_name]:.2f}',
            f'{fix_med:.2f}',
            f'{meds[v_name]:.2f}',
            f'{ratio_vs_cpu:.2f}',
            f'{ratio_vs_cuda:.2f}',
        ])

    print(tabulate(
        fix_rows,
        headers=['variant', 'CPU', 'CPU+xfer', 'CUDA', 'vs CPU', 'vs CUDA'],
        tablefmt='simple',
    ))
    print('\n  vs CPU  = transfer overhead on top of CPU compute  (ideally near 1.0)')
    print('  vs CUDA = speedup of the fix vs. running GLMM on CUDA  (ideally < 1.0)')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-id',    default='small-n-sampled')
    parser.add_argument('--partition',  default='valid')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batches',    type=int, default=20,
                        help='number of batches to time (excl. warmup)')
    parser.add_argument('--device',     default='cpu',
                        help='cpu or cuda (or cuda:N for a specific GPU)')
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run(setup())
