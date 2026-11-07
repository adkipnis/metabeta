"""Group budget on simulated data: nested cells through the checkpoints' limit point.

The checkpoints were trained on at most 200 groups and 3000 rows (the ``standard`` shape
profile of presets.yaml), and the Api refuses data beyond either. This script writes test sets
whose groups and rows vary on independent axes through that point, groups {100, 200, 250, 400}
at 3000 rows and rows {1500, 3000, 3500, 6000} at 200 groups, plus the corner (400, 6000),
for d=4 and q=1 (the ``small`` regime). Rows per group stay within the trained 5-150, so only
the group count and the row total leave the budget.

Each master dataset is one unchanged draw of the training simulator (hyperprior -> prior ->
parameters, Synthesizer design, Simulator.sample) with 400 groups of 30 rows. A cell (m, n)
takes the first m groups and the first n // m rows of each (the first n % m groups one row
more), so the cells of a master share its prior, its parameters (the effects of the shared
groups included) and its standardization (X, and sd_y for Gaussian outcomes): they differ
only in the data added or removed. Synthesizer rows are i.i.d. within and across groups, so
a group's first rows are a random subset of its rows.

A master is redrawn when a cell would hold a constant predictor, or, for Poisson outcomes,
when the true linear predictor exceeds the simulator's clip in any row: utils.evaluation.
simulatorClipped drops such datasets from the oracle tables but not from the real-data
tables, so the two would score different datasets.

Usage from repo root:
    uv run python experiments/simulation/group_budget_sets.py --families n --n_datasets 2 --dry_run
    uv run python experiments/simulation/group_budget_sets.py
    uv run python experiments/simulation/group_budget_sets.py --print_commands
"""

import argparse
import logging

import numpy as np
import yaml

from metabeta.simulation import Prior, Simulator, Synthesizer, hypersample
from metabeta.simulation.generate import Generator
from metabeta.simulation.simulator import linearPredictor
from metabeta.utils.constants import FAMILY_NAMES_REVERSE
from metabeta.utils.experiments import DATA_DIR
from metabeta.utils.families import POISSON_ETA_CLIP_MAX
from metabeta.utils.logger import setupLogging
from metabeta.utils.padding import aggregate
from metabeta.utils.templates import generateSimulationConfig

logger = logging.getLogger(__name__)

SIZE = 'small'
D, Q = 4, 1
MASTER_M, MASTER_NI = 400, 30
# (groups, rows); the first three lie within the budget of 200 groups and 3000 rows
CELLS = [
    (100, 3000),
    (200, 3000),
    (200, 1500),
    (250, 3000),
    (400, 3000),
    (200, 3500),
    (200, 6000),
    (400, 6000),
]
MAX_REDRAWS = 50


def cellTag(m: int, n: int) -> str:
    return f'gb{m}x{n}'


def cellIds(family: str) -> list[str]:
    return [f'{SIZE}-{family}-{cellTag(m, n)}' for m, n in CELLS]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate the nested group-budget test sets.')
    parser.add_argument('--families', nargs='+', default=['n', 'b', 'p'], choices=['n', 'b', 'p'])
    parser.add_argument('--n_datasets', type=int, default=16, help='master datasets per family')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true', help='generate and check, write nothing')
    parser.add_argument('--print_commands', action='store_true', help='only print the fit campaign')
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def cellNs(m: int, n: int) -> np.ndarray:
    ns = np.full(m, n // m)
    ns[: n % m] += 1
    return ns


def cellRows(ns: np.ndarray) -> np.ndarray:
    """Row indices of the master taking the first ns[g] rows of group g."""
    return np.concatenate([g * MASTER_NI + np.arange(k) for g, k in enumerate(ns)])


def cell(master: dict[str, np.ndarray], m: int, n: int) -> dict[str, np.ndarray]:
    ns = cellNs(m, n)
    rows = cellRows(ns)
    out = dict(master)
    out.update(
        X=master['X'][rows],
        y=master['y'][rows],
        groups=master['groups'][rows],
        rfx=master['rfx'][:m],
        ns=ns,
        m=np.array(m),
        n=np.array(n),
    )
    return out


def masterProblem(master: dict[str, np.ndarray], lf: int) -> str | None:
    """Why a master must be redrawn, or None."""
    for m, n in CELLS:
        x = cell(master, m, n)['X'][:, 1:]
        if np.any(x.std(0) == 0):
            return f'constant predictor in cell ({m}, {n})'
    if lf == 2:
        eta = linearPredictor(master, master)
        if eta.max() > POISSON_ETA_CLIP_MAX:
            return f'true eta {eta.max():.2f} exceeds the Poisson clip'
    return None


def drawMaster(seedseq: np.random.SeedSequence, lf: int) -> tuple[dict[str, np.ndarray], int]:
    """One master dataset and the number of redraws it took."""
    for attempt, seed in enumerate(seedseq.spawn(MAX_REDRAWS)):
        rng = np.random.default_rng(seed)
        prior = Prior(rng, hypersample(rng, D, Q, likelihood_family=lf))
        ns = np.full(MASTER_M, MASTER_NI)
        master = Simulator(rng, prior, Synthesizer(rng), ns).sample()
        assert np.array_equal(master['groups'], np.repeat(np.arange(MASTER_M), MASTER_NI))
        problem = masterProblem(master, lf)
        if problem is None:
            return master, attempt
        logger.info('  redraw (%s)', problem)
    raise RuntimeError(f'no valid master after {MAX_REDRAWS} draws')


def writeCell(family: str, m: int, n: int, cells: list[dict], cfg: argparse.Namespace) -> None:
    data_id = f'{SIZE}-{family}-{cellTag(m, n)}'
    out_dir = DATA_DIR / data_id
    path = out_dir / 'test.npz'
    if path.exists():
        raise FileExistsError(f'{path} exists; this script never overwrites a test set')
    batch = Generator._castCompactTypes(aggregate(cells))
    config = generateSimulationConfig(
        size=SIZE, family=FAMILY_NAMES_REVERSE[family], ds_type=cellTag(m, n)
    )
    config.update(
        data_id=data_id,
        bs_test=len(cells),
        group_budget_cell=[m, n],
        group_budget_master=[MASTER_M, MASTER_NI],
        group_budget_seed=cfg.seed,
    )
    out_dir.mkdir(parents=True)
    np.savez_compressed(path, **batch, allow_pickle=True)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info('%s: %d datasets -> %s', data_id, len(cells), path)


def generateFamily(family: str, cfg: argparse.Namespace) -> None:
    lf = FAMILY_NAMES_REVERSE[family]
    root = np.random.SeedSequence([cfg.seed, lf])
    masters = []
    for i, seedseq in enumerate(root.spawn(cfg.n_datasets)):
        master, redraws = drawMaster(seedseq, lf)
        logger.info('%s master %d: %d redraw(s)', family, i, redraws)
        masters.append(master)
    for m, n in CELLS:
        cells = [cell(master, m, n) for master in masters]
        if cfg.dry_run:
            logger.info('%s cell (%d, %d): ok (dry run)', family, m, n)
            continue
        writeCell(family, m, n, cells, cfg)


def printCommands(families: list[str], n_datasets: int) -> None:
    ids = [data_id for family in families for data_id in cellIds(family)]
    print('# analytical statistics first: they are written into test.npz, which fits checksum')
    for data_id in ids:
        family = FAMILY_NAMES_REVERSE[data_id.split('-')[1]]
        tag = data_id.split('-', 2)[2]
        print(
            f'python metabeta/analytical/precompute.py --size {SIZE} --family {family} '
            f'--ds_type {tag} --partition test'
        )
    print('# reference fits')
    for data_id in ids:
        for method, level in (('nuts', 2), ('advi', 1), ('pathfinder', 1)):
            print(
                f'scripts/fit-ref.sh --method {method} --level {level} '
                f'--n_datasets {n_datasets} --data_id {data_id}'
            )
    print('# after the fits finished (checks the fits, then reintegrates)')
    for data_id in ids:
        print(f'python metabeta/simulation/check.py --partition test --data_id {data_id}')


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    if cfg.print_commands:
        printCommands(cfg.families, cfg.n_datasets)
        return
    for family in cfg.families:
        generateFamily(family, cfg)
    if not cfg.dry_run:
        printCommands(cfg.families, cfg.n_datasets)


if __name__ == '__main__':
    main()
