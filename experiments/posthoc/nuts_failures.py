"""Failure rate of Api.nuts at its defaults (#72).

Per cell (family, size), Api.sample with its defaults on each test dataset, then Api.nuts with its
defaults, through a joint checkpoint that holds only the cell's network. Both calls take the
dataset's index as their seed. Each fit appends one row to
experiments/results/nuts-failures/{family}_{size}.csv: the dataset, the flags of sample whose
warnings name Api.nuts, the NUTS diagnostics and the seconds of the nuts call. A rerun skips the
datasets already written, so a task that hits its time limit resumes.

A fit fails at max R-hat > 1.05 or min bulk ESS < 100, the experiments' cold-rerun rule
(posthoc/warmnuts.py, needsEscalation); --table prints the rates per cell.

Usage (from repo root):
    uv run python experiments/posthoc/nuts_failures.py --cell 0 --n-datasets 2
    uv run python experiments/posthoc/nuts_failures.py --table
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

import metabeta
from metabeta.models.api import Api
from metabeta.posthoc.warmnuts import needsEscalation
from metabeta.utils.api import joinCheckpoints
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, experimentResultsPath

# the checkpoint-seed mapping of the published joint checkpoints
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, FAMILY_INITIAL, _ckpt_dir  # noqa: E402

# ==============================================================================
OUT_DIR = experimentResultsPath('nuts-failures')
COLUMNS = ('dataset', 'flags', 'rhat_max', 'ess_bulk_min', 'ess_tail_min', 'divergences', 'seconds')

# the safeguards whose warnings name Api.nuts (models/api.py, _warnSafeguards)
NUTS_FLAGS = ('accept', 'ess', 'prior', 'size', 'map_z')

# (family, size, cold reruns out of 512): the cells whose warmNuts logs count the cold reruns at
# 500 warmup steps and 500 draws, in experiments/results/ablation-refresh/{family}/
# log_{size}_warmNuts.txt and experiments/results/clip-excl/logs/ablation-{size}-*.txt
CELLS = (
    ('normal', 'small', 62),
    ('normal', 'medium', 99),
    ('bernoulli', 'small', 18),
    ('bernoulli', 'medium', 41),
    ('bernoulli', 'large', 47),
    ('poisson', 'medium', 62),
    ('poisson', 'large', 81),
    ('poisson', 'huge', 114),
)


# ==============================================================================
class NutsFailures:
    """Api.sample, then Api.nuts, on each test dataset of one cell, one CSV row per fit."""

    def __init__(self, cell: int, prefix: str, n_datasets: int | None) -> None:
        self.family, self.size, _ = CELLS[cell]
        self.prefix = prefix
        data_id = f'{self.size}-{FAMILY_INITIAL[self.family]}-sampled'
        self.collection = Collection(DATA_DIR / data_id / 'test.npz', permute=False)
        n = len(self.collection) if n_datasets is None else n_datasets
        self.out = OUT_DIR / f'{self.family}_{self.size}.csv'
        done = set(pd.read_csv(self.out)['dataset']) if self.out.exists() else set()
        self.todo = [i for i in range(n) if i not in done]

    def loadApi(self) -> None:
        ckpt_dir = _ckpt_dir(self.family, self.size, BEST_SEEDS[(self.family, self.size)])
        with tempfile.TemporaryDirectory() as tmp:
            joint = joinCheckpoints(
                {f'{self.family}-{self.size}': ckpt_dir},
                Path(tmp) / 'joint.pt',
                prefixes=self.prefix,
            )
            self.api = Api(joint, device='cpu')

    def fit(self, i: int) -> dict:
        result = self.api.sample(collateGrouped([self.collection[i]]), seed=i)
        flags = [f for f in NUTS_FLAGS if result.safeguards[f'{f}_flag'][0]]
        start = time.perf_counter()
        sg = self.api.nuts(result, seed=i).safeguards
        return {
            'dataset': i,
            'flags': ' '.join(flags),
            'rhat_max': float(sg['rhat_max'][0]),
            'ess_bulk_min': float(sg['ess_bulk_min'][0]),
            'ess_tail_min': float(sg['ess_tail_min'][0]),
            'divergences': int(sg['divergences'][0]),
            'seconds': round(time.perf_counter() - start, 2),
        }

    def go(self) -> None:
        print(
            f'{self.family} {self.size}: {len(self.todo)} datasets to fit with metabeta from '
            f'{Path(metabeta.__file__).parent}',
            flush=True,
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        header = not self.out.exists()
        with self.out.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if header:
                writer.writeheader()
            for i in self.todo:
                writer.writerow(self.fit(i))
                f.flush()


def failureTable() -> str:
    """Per cell: the failure rate over all fits and over those sample flags, the mean seconds of
    a nuts call and the logged cold reruns."""

    def rate(failed: pd.Series) -> str:
        return f'{failed.mean():.1%} ({failed.sum()}/{len(failed)})' if len(failed) else '-'

    lines = [
        '| cell | failed, all | failed, flagged by sample | seconds per fit '
        '| cold reruns at 500/500 |',
        '|---|---|---|---|---|',
    ]
    for family, size, reruns in CELLS:
        path = OUT_DIR / f'{family}_{size}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        failed = df.apply(lambda r: needsEscalation(r.rhat_max, r.ess_bulk_min), axis=1)
        flagged = df['flags'].fillna('') != ''
        lines.append(
            f'| {family} {size} | {rate(failed)} | {rate(failed[flagged])} '
            f'| {df["seconds"].mean():.1f} | {reruns / 512:.1%} ({reruns}/512) |'
        )
    return '\n'.join(lines)


# ==============================================================================
# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cell', type=int, choices=range(len(CELLS)), help='index into CELLS (one SLURM array task per cell)')
    p.add_argument('--prefix', type=str, default='latest', help='checkpoint prefix (latest = the checkpoint of the oracle tables)')
    p.add_argument('--n-datasets', type=int, default=None, help='fit only the first n test datasets (default: all)')
    p.add_argument('--table', action='store_true', help='print the failure table from the CSVs written so far')
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    args = setup()
    if args.table:
        print(failureTable())
    else:
        run = NutsFailures(args.cell, args.prefix, args.n_datasets)
        run.loadApi()
        run.go()
