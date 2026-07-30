"""Calibrate the MAP z-score safeguard threshold from cached metabeta samples.

The API safeguard (metabeta/models/api.py: mapZScores) flags datasets whose posterior
mean drifts too far from the analytical MAP anchor, in posterior-SD units.  This script
computes the per-dataset max-z statistic on held-out data — where the posterior is known
to be well calibrated — and reports its distribution, so the warning threshold
(api.MAP_Z_THRESHOLD) can be set to a high in-distribution quantile (e.g. 99%) instead
of a guess.

Optionally pass misspecified condition dirs (e.g. --ds_types sampled student3 xt1) to
check that the flag rate rises under distribution shift; condition dirs need analytical
stats precomputed (metabeta/analytical/precompute.py) and MB sample caches are created
on first run.  Cached samples and stats live in standardized outcome space, matching the
space in which the API computes the check — nothing is rescaled here.

Usage (from repo root):
    uv run python experiments/evaluation/map_z_calibration.py --families n --sizes small
    uv run python experiments/evaluation/map_z_calibration.py --families n b p
    uv run python experiments/evaluation/map_z_calibration.py --families n --ds_types sampled student3
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.models.api import mapZScores
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.sampling import setSeed
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, REPO_ROOT
from metabeta.utils.posterior_eval import loadModel, loadOrSampleMB

sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

logger = logging.getLogger(__name__)

FAMILY_NAMES = {'n': 'normal', 'b': 'bernoulli', 'p': 'poisson'}
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']
QUANTILES = [0.50, 0.90, 0.95, 0.99, 0.999]
CANDIDATE_THRESHOLDS = [3.0, 4.0, 5.0, 7.0, 10.0]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calibrate the MAP z-score safeguard threshold.')
    parser.add_argument('--families', nargs='+', default=['n'], choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--ds_types', nargs='+', default=['sampled'], help='data dir tags; the first is treated as the in-distribution reference')
    parser.add_argument('--partition', type=str, default='test', choices=['test', 'valid'])
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def collectZ(
    cfg: argparse.Namespace,
    family: str,
    size: str,
    ds_type: str,
    device: torch.device,
) -> np.ndarray | None:
    """Per-dataset max-z scores for one data dir, from cached MB samples where available."""
    data_id = f'{size}-{family}-{ds_type}'
    seed = BEST_SEEDS.get((FAMILY_NAMES[family], size))
    if seed is None:
        logger.warning('%s: no BEST_SEEDS checkpoint — skipping', data_id)
        return None
    ckpt_dir = _ckpt_dir(FAMILY_NAMES[family], size, seed)
    data_path = DATA_DIR / data_id / f'{cfg.partition}.fit.npz'
    if not data_path.exists():
        data_path = DATA_DIR / data_id / f'{cfg.partition}.npz'
    if not data_path.exists() or not ckpt_dir.exists():
        logger.warning('%s: data or checkpoint missing — skipping', data_id)
        return None

    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)
    col = Collection(data_path, permute=False, max_d=model_cfg.max_d, max_q=model_cfg.max_q)
    batch = collateGrouped([col[i] for i in range(len(col))])
    if 'stats' not in batch:
        # fit files may lack the analytical stats; fall back to the sibling npz
        base_path = data_path.with_name(f'{cfg.partition}.npz')
        if base_path.exists():
            base_col = Collection(
                base_path, permute=False, max_d=model_cfg.max_d, max_q=model_cfg.max_q
            )
            if len(base_col) == len(col):
                base_batch = collateGrouped([base_col[i] for i in range(len(base_col))])
                if 'stats' in base_batch:
                    batch['stats'] = base_batch['stats']
    if 'stats' not in batch:
        logger.warning(
            '%s: no analytical stats in %s — run metabeta/analytical/precompute.py first',
            data_id,
            data_path.name,
        )
        return None

    proposal, _ = loadOrSampleMB(
        model,
        batch,
        data_path,
        ckpt_dir,
        cfg.prefix,
        cfg.n_samples,
        cfg.batch_size,
        cfg.seed,
        device,
        None,
    )
    z = mapZScores(proposal, batch['stats'], batch)
    logger.info(
        '%s: %d datasets, median z = %.2f, max z = %.2f', data_id, len(z), np.median(z), z.max()
    )
    return z


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for family in cfg.families:
        for ds_type in cfg.ds_types:
            pooled = [
                z
                for z in (collectZ(cfg, family, s, ds_type, device) for s in cfg.sizes)
                if z is not None
            ]
            if not pooled:
                continue
            z = np.concatenate(pooled)
            row = {'family': FAMILY_NAMES[family], 'ds_type': ds_type, 'n': len(z)}
            for q in QUANTILES:
                row[f'q{q:g}'] = float(np.quantile(z, q))
            for t in CANDIDATE_THRESHOLDS:
                row[f'flag@{t:g}'] = float((z > t).mean())
            rows.append(row)

    if not rows:
        logger.error('No data collected.')
        return

    headers = (
        ['family', 'ds_type', 'n']
        + [f'z @ {q:g}' for q in QUANTILES]
        + [f'flag% @ {t:g}' for t in CANDIDATE_THRESHOLDS]
    )
    table = [
        [r['family'], r['ds_type'], r['n']]
        + [f"{r[f'q{q:g}']:.2f}" for q in QUANTILES]
        + [f"{100 * r[f'flag@{t:g}']:.1f}" for t in CANDIDATE_THRESHOLDS]
        for r in rows
    ]
    md = tabulate(table, headers=headers, tablefmt='pipe', stralign='right')
    print('\n=== MAP z-score distribution (per-dataset max-z) ===\n')
    print(md)
    print(
        '\nPick api.MAP_Z_THRESHOLD near the in-distribution 99% quantile; flag rates on '
        'misspecified ds_types show the detection power at that threshold.'
    )

    out_path = outdir / 'map_z_calibration.md'
    out_path.write_text(
        '# MAP z-score calibration\n\n'
        f'Partition: {cfg.partition}. Sizes pooled: {", ".join(cfg.sizes)}. '
        f'{cfg.n_samples} posterior samples per dataset.\n\n' + md + '\n'
    )
    logger.info('Saved table to %s', out_path)


if __name__ == '__main__':
    main()
