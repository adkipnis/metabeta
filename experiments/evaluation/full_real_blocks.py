"""Agreement with NUTS per parameter block on the whole real datasets beyond the routing limits,
beside within-budget datasets of the same family and size regime.

real_posterior.py pools r, σ-ratio and rank-MAD over all active entries of a dataset, so on the
whole sets (thousands of groups) they are group-effect statistics. This script reads the
per-dataset output of real_posterior.py (``real_<run>_<data_id>_per_dataset.npz``) and reports
each metric per block (ffx, sigma_rfx, sigma_eps, rfx) as median ± MAD over datasets, the
statistic of the paper's real-data tables, for these populations:

  whole      the whole dataset (one dataset, no spread)
  regime     every NUTS-converged dataset of the within-budget set (the paper's table row)
  regime q1  the same with one random effect, as in the whole sets
  source     the within-budget subsamples of the same source dataset
  source q1  the same with one random effect

Usage from repo root, after ``scripts/full-real.sh compare`` and ``within``:
    python experiments/evaluation/full_real_blocks.py \
        --results experiments/results/full-real experiments/results/full-real/within
"""

import argparse
from pathlib import Path

import numpy as np
from tabulate import tabulate

from metabeta.utils.experiments import DATA_DIR

from real_posterior import _medianMad

# whole set, within-budget set, source stem of the subsamples (None: never drawn), checkpoint
SETS = [
    (
        'large-n-full',
        'large-n-real',
        'insteval__grp_group',
        'data=large-n-mixed_model=large_seed=9',
    ),
    ('medium-b-full', 'medium-b-real', None, 'data=medium-b-mixed_model=large_seed=3'),
    ('small-p-full', 'small-p-real', 'chem97__grp_group', 'data=small-p-mixed_model=large_seed=4'),
]
POOLED = ('r', 'sigma_ratio', 'rank_mad', 'delta_nll')
BLOCKS = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
BLOCK_METRICS = ('sigma_ratio', 'rank_mad', 'r')


def populations(data_id: str, whole: str, source: str | None) -> dict[str, tuple[str, np.ndarray]]:
    """Population name -> (data id, boolean mask over that test.npz)."""
    meta = np.load(DATA_DIR / data_id / 'test.npz', allow_pickle=True)
    q1 = meta['q'] == 1
    out = {'whole': (whole, np.ones(1, dtype=bool)), 'regime': (data_id, np.ones_like(q1))}
    out['regime q1'] = (data_id, q1)
    if source is not None:
        from_source = np.array([str(np.asarray(s).item()) == source for s in meta['source']])
        out['source'] = (data_id, from_source)
        out['source q1'] = (data_id, from_source & q1)
    return out


def methods(npz) -> list[str]:
    return list(dict.fromkeys(k.split('/')[0] for k in npz.files))


def cell(values: np.ndarray, dp: int = 2) -> str:
    if len(values) == 1:
        return 'NA' if np.isnan(values[0]) else f'{values[0]:.{dp}f}'
    med, mad = _medianMad(values)
    return 'NA' if med != med else f'{med:.{dp}f} ± {mad:.{dp}f}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        type=Path,
        nargs='+',
        required=True,
        help='Directories holding the per-dataset npz files; the first gets the report',
    )
    parser.add_argument('--prefix', type=str, default='latest')
    cfg = parser.parse_args()

    tables: dict[str, list[list[str]]] = {'pooled': []} | {m: [] for m in BLOCK_METRICS}
    for whole, within, source, ckpt in SETS:
        for pop, (data_id, mask) in populations(within, whole, source).items():
            name = f'real_{ckpt}_{cfg.prefix}_{data_id}_per_dataset.npz'
            (path,) = [d / name for d in cfg.results if (d / name).exists()]
            npz = np.load(path)
            meta = np.load(DATA_DIR / data_id / 'test.npz', allow_pickle=True)
            n_rfx = meta['m'] * meta['q']
            for method in methods(npz):
                index = npz[f'{method}/index']
                keep = mask[index]
                # share of the pooled entries that are group effects
                n_global = meta['d'] + meta['q'] + int(f'{method}/sigma_eps/r' in npz.files)
                share = (n_rfx / (n_rfx + n_global))[index][keep]
                head = [whole, pop, method, str(int(keep.sum()))]
                tables['pooled'].append(
                    head + [cell(npz[f'{method}/{k}'][keep]) for k in POOLED] + [cell(share)]
                )
                for metric in BLOCK_METRICS:
                    tables[metric].append(
                        head
                        + [
                            cell(npz[f'{method}/{b}/{metric}'][keep])
                            if f'{method}/{b}/{metric}' in npz.files
                            else ''
                            for b in BLOCKS
                        ]
                    )

    parts = ['# Agreement with nuts2 per parameter block, median ± MAD over datasets']
    head = ['set', 'population', 'method', 'N']
    parts.append(
        '## Pooled over all active entries (real_posterior.py)\n\n'
        + tabulate(tables['pooled'], headers=head + list(POOLED) + ['rfx share'], tablefmt='pipe')
    )
    for metric in BLOCK_METRICS:
        parts.append(
            f'## {metric} per block\n\n'
            + tabulate(tables[metric], headers=head + list(BLOCKS), tablefmt='pipe')
        )
    out = cfg.results[0] / f'blocks_{cfg.prefix}.md'
    out.write_text('\n\n'.join(parts) + '\n')
    print('\n\n'.join(parts))
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
