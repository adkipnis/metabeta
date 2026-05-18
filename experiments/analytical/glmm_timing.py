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

CONTEXTS = ['no_grad', 'grad']


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


def _run_batch(
    batch: dict[str, torch.Tensor],
    max_q: int,
    likelihood_family: int,
    variant_kwargs: dict,
    device: torch.device,
) -> float:
    """Run glmm on one batch, return wall seconds.

    CUDA operations are asynchronous: without synchronize() the clock would
    only capture kernel dispatch time, not actual execution time.
    """
    Zm = batch['Z'][..., :max_q]
    common = _build_common(batch, max_q)
    _sync(device)
    t0 = time.perf_counter()
    glmm(
        batch['X'],
        batch['y'],
        Zm,
        batch['mask_n'].float(),
        batch['mask_m'].float(),
        batch['ns'].clamp(min=1).float(),
        batch['n'].float(),
        likelihood_family=likelihood_family,
        **common,
        **variant_kwargs,
    )
    _sync(device)
    return time.perf_counter() - t0


def _time_variant(
    batches: list[dict[str, torch.Tensor]],
    max_q: int,
    likelihood_family: int,
    variant_kwargs: dict,
    context: str,
    device: torch.device,
    warmup: int,
) -> tuple[float, float]:
    """Return (median_ms_per_ds, std_ms_per_ds) over batches."""
    times_ms: list[float] = []

    def _step(b: dict) -> float:
        return _run_batch(b, max_q, likelihood_family, variant_kwargs, device)

    for i, batch in enumerate(batches):
        if context == 'no_grad':
            with torch.no_grad():
                elapsed = _step(batch)
        else:
            elapsed = _step(batch)

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

    # collect results
    rows = []
    for v_name, v_kwargs in VARIANTS:
        row = [v_name]
        for ctx in CONTEXTS:
            med, std = _time_variant(
                batches, max_q, likelihood_family, v_kwargs, ctx, device, warmup,
            )
            row += [f'{med:.2f}', f'{std:.2f}']
        rows.append(row)

    headers = ['variant', 'no_grad med', 'no_grad std', 'grad med', 'grad std',]
    print(tabulate(rows, headers=headers, tablefmt='simple'))

    # ratio table: grad / no_grad median
    print('\ngrad / no_grad ratio  (1.0 = outer grad context adds no overhead):')
    ratio_rows = []
    for row in rows:
        v_name = row[0]
        ng_med = float(row[1])
        g_med  = float(row[3])
        ratio  = g_med / ng_med if ng_med > 0 else float('nan')
        ratio_rows.append([v_name, f'{ng_med:.2f}', f'{g_med:.2f}', f'{ratio:.3f}'])
    print(tabulate(ratio_rows, headers=['variant', 'no_grad ms/ds', 'grad ms/ds', 'ratio'],
                   tablefmt='simple'))

    # step-count scaling
    print('\nIncremental cost per MAP step  (no_grad context):')
    ng_meds = {row[0]: float(row[1]) for row in rows}
    raw_t  = ng_meds['raw']
    m5_t   = ng_meds['map_5']
    m10_t  = ng_meds['map_10']
    m20_t  = ng_meds['map_20']
    inc_rows = [
        ['raw → map_5  (+5 steps)',  f'{m5_t  - raw_t:.2f}', f'{(m5_t  - raw_t)/5:.3f}'],
        ['map_5 → map_10 (+5)',      f'{m10_t - m5_t:.2f}',  f'{(m10_t - m5_t)/5:.3f}'],
        ['map_10 → map_20 (+10)',    f'{m20_t - m10_t:.2f}', f'{(m20_t - m10_t)/10:.3f}'],
    ]
    print(tabulate(inc_rows, headers=['interval', 'delta ms/ds', 'ms/ds per step'],
                   tablefmt='simple'))

    eb_overhead = ng_meds['default'] - ng_meds['map_20']
    print(f'\nLaplaceEB overhead on top of map_20  (no_grad): {eb_overhead:.2f} ms/ds')
    print(f'LaplaceEB alone (eb_only - raw)                : {ng_meds["eb_only"] - raw_t:.2f} ms/ds')
    print(f'\ndefault vs raw  speedup factor: {ng_meds["default"] / raw_t:.1f}x')


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
