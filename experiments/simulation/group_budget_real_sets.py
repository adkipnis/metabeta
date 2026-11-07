"""Group budget on real data: the nested cells of group_budget_sets.py on three real sources.

The cells, d=4, q=1, the `small` checkpoints and the training-hyperprior priors are those of the
simulated sets; only the data differ. Each family has one real source:

- n: stroop (afex), log RT on condition, congruency and their product, by participant;
- b: Yelp (BeyondMLR), whether a review was voted useful, 3 of 5 covariates, by restaurant;
- p: Chem97 (mlmRev), A-level score on the 3 predictors of the paper's held-out set, by school.

stroop and Yelp come from group_budget_real_export.py, Chem97 from metabeta's preprocessed
held-out file. Each master holds 400 groups drawn at random, the first 200 from the groups with
>= 30 rows and the next 200 from the rest with >= 15, each group's rows shuffled and cut to 30
and 15. A cell (m, n) takes the first m groups and n // m rows of each (the first n % m groups
one row more), as in the simulated sets, so every group of a cell has the same row count.
Predictors are transformed and Gaussian outcomes scaled on the master as Subsampler does on a
subsample, so the cells of a master share its standardization and its prior.

Rows are selected here, not with Subsampler, whose labels do not follow its rows
(Subsampler._subsetGrouped keeps rows in source order and labels them in member order). Each
cell's source row indices go to source_rows.npz beside its test.npz, and every label is checked
to cover exactly one source group before anything is written.

Usage from repo root:
    python experiments/simulation/group_budget_real_sets.py --dry_run
    python experiments/simulation/group_budget_real_sets.py
    python experiments/simulation/group_budget_real_sets.py --print_commands
"""

import argparse
import logging

import numpy as np
import yaml
from group_budget_sets import CELLS, SIZE, D, Q, cellNs, printCommands

from metabeta.simulation import hypersample
from metabeta.simulation.generate import Generator
from metabeta.utils.constants import FAMILY_NAMES_REVERSE
from metabeta.utils.experiments import DATA_DIR, PREPROCESSED_DATA_DIR
from metabeta.utils.families import hasSigmaEps
from metabeta.utils.logger import setupLogging
from metabeta.utils.padding import aggregate
from metabeta.utils.preprocessing import transformPredictors
from metabeta.utils.templates import generateSimulationConfig

logger = logging.getLogger(__name__)

SOURCE_DIR = DATA_DIR / 'group-budget-sources'
SOURCES = {
    'n': SOURCE_DIR / 'stroop.npz',
    'b': SOURCE_DIR / 'yelp.npz',
    'p': PREPROCESSED_DATA_DIR / 'test' / 'chem97__grp_group.npz',
}
MASTER_NS = np.repeat([30, 15], 200)
MAX_REDRAWS = 50


def cellTag(m: int, n: int) -> str:
    return f'gbr{m}x{n}'


def cellIds(family: str) -> list[str]:
    return [f'{SIZE}-{family}-{cellTag(m, n)}' for m, n in CELLS]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate the nested real-data group-budget sets.')
    parser.add_argument('--families', nargs='+', default=['n', 'b', 'p'], choices=['n', 'b', 'p'])
    parser.add_argument('--n_datasets', type=int, default=16, help='masters per family')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true', help='generate and check, write nothing')
    parser.add_argument('--print_commands', action='store_true', help='only print the fit campaign')
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def loadSource(family: str) -> dict[str, np.ndarray]:
    src = dict(np.load(SOURCES[family], allow_pickle=True))
    assert np.all(np.diff(src['groups']) >= 0), 'source groups must be sorted'
    src['start'] = np.concatenate([[0], np.cumsum(src['ns'])[:-1]])
    src['name'] = SOURCES[family].stem
    return src


def cellRows(ns: np.ndarray) -> np.ndarray:
    """Master row indices taking the first ns[g] rows of master group g."""
    start = np.concatenate([[0], np.cumsum(MASTER_NS)[:-1]])
    return np.concatenate([start[g] + np.arange(k) for g, k in enumerate(ns)])


def cell(master: dict[str, np.ndarray], m: int, n: int) -> dict[str, np.ndarray]:
    ns = cellNs(m, n)
    assert np.all(ns <= MASTER_NS[:m])
    rows = cellRows(ns)
    out = dict(master)
    out.update(
        X=master['X'][rows],
        y=master['y'][rows],
        groups=master['groups'][rows],
        source_rows=master['source_rows'][rows],
        rfx=master['rfx'][:m],
        ns=ns,
        m=np.array(m),
        n=np.array(n),
    )
    return out


def sourceRows(rng: np.random.Generator, src: dict[str, np.ndarray]) -> np.ndarray:
    """Source row indices of a master: 400 random groups, each its rows shuffled and cut."""
    first = rng.choice(np.flatnonzero(src['ns'] >= 30), 200, replace=False)
    rest = np.setdiff1d(np.flatnonzero(src['ns'] >= 15), first)
    members = np.concatenate([first, rng.choice(rest, 200, replace=False)])
    return np.concatenate(
        [src['start'][g] + rng.permutation(src['ns'][g])[:k] for g, k in zip(members, MASTER_NS)]
    )


def drawMaster(
    seedseq: np.random.SeedSequence, src: dict[str, np.ndarray], lf: int
) -> tuple[dict[str, np.ndarray], int]:
    """One master dataset and the number of redraws it took."""
    for attempt, seed in enumerate(seedseq.spawn(MAX_REDRAWS)):
        rng = np.random.default_rng(seed)
        n_cols = src['X'].shape[1]
        cols = np.sort(rng.choice(n_cols, D - 1, replace=False))
        rows = sourceRows(rng, src)
        x = src['X'][rows][:, cols]
        constant = [(m, n) for m, n in CELLS if np.any(x[cellRows(cellNs(m, n))].std(0) == 0)]
        if constant:
            logger.info('  redraw (constant predictor in cells %s)', constant)
            continue
        x = transformPredictors(x, axis=0, exclude_binary=True, transform_counts=True)
        x = np.concatenate([np.ones((len(x), 1)), x], axis=-1)
        y = src['y'][rows]
        sd_y = max(float(y.std()), 1e-6) if lf == 0 else 1.0
        if lf == 0:
            y = y / sd_y
        master = {
            'ffx': np.full(D, np.nan),
            'sigma_rfx': np.full(Q, np.nan),
            'corr_rfx': np.full((Q, Q), np.nan),
            'rfx': np.full((len(MASTER_NS), Q), np.nan),
            **{k: np.asarray(v) for k, v in hypersample(rng, D, Q, likelihood_family=lf).items()},
            'y': y,
            'X': x,
            'groups': np.repeat(np.arange(len(MASTER_NS)), MASTER_NS),
            'm': np.array(len(MASTER_NS)),
            'n': np.array(len(rows)),
            'ns': MASTER_NS,
            'd': np.array(D),
            'q': np.array(Q),
            'sd_y': np.array(sd_y),
            'source': np.array(src['name'], dtype=object),
            'source_rows': rows,
            'source_cols': cols,
        }
        if hasSigmaEps(lf):
            master['sigma_eps'] = np.array(np.nan)
            master['r_squared'] = np.array(np.nan)
        return master, attempt
    raise RuntimeError(f'no valid master after {MAX_REDRAWS} draws')


def checkLabels(cells: list[dict[str, np.ndarray]], src: dict[str, np.ndarray]) -> None:
    """Every label of a cell covers exactly one source group, and no source group twice."""
    for c in cells:
        pairs = np.unique(np.stack([c['groups'], src['groups'][c['source_rows']]]), axis=1)
        assert pairs.shape[1] == int(c['m']), 'a label spans several source groups'
        assert len(np.unique(pairs[1])) == int(c['m']), 'a source group carries several labels'
        assert np.array_equal(np.bincount(c['groups']), c['ns'])


def writeCell(family: str, m: int, n: int, cells: list[dict], cfg: argparse.Namespace) -> None:
    data_id = f'{SIZE}-{family}-{cellTag(m, n)}'
    out_dir = DATA_DIR / data_id
    path = out_dir / 'test.npz'
    if path.exists():
        raise FileExistsError(f'{path} exists; this script never overwrites a test set')
    side = {k: np.stack([c.pop(k) for c in cells]) for k in ('source_rows', 'source_cols')}
    batch = Generator._castCompactTypes(aggregate(cells))
    config = generateSimulationConfig(
        size=SIZE, family=FAMILY_NAMES_REVERSE[family], ds_type=cellTag(m, n)
    )
    config.update(
        data_id=data_id,
        bs_test=len(cells),
        group_budget_cell=[m, n],
        group_budget_master_ns=[[200, 30], [200, 15]],
        group_budget_source=SOURCES[family].name,
        group_budget_seed=cfg.seed,
    )
    out_dir.mkdir(parents=True)
    np.savez_compressed(path, **batch, allow_pickle=True)
    np.savez_compressed(out_dir / 'source_rows.npz', **side, source=SOURCES[family].name)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info('%s: %d datasets -> %s', data_id, len(cells), path)


def generateFamily(family: str, cfg: argparse.Namespace) -> None:
    lf = FAMILY_NAMES_REVERSE[family]
    src = loadSource(family)
    root = np.random.SeedSequence([cfg.seed, lf])
    masters = []
    for i, seedseq in enumerate(root.spawn(cfg.n_datasets)):
        master, redraws = drawMaster(seedseq, src, lf)
        logger.info('%s master %d: %d redraw(s)', family, i, redraws)
        masters.append(master)
    for m, n in CELLS:
        cells = [cell(master, m, n) for master in masters]
        checkLabels(cells, src)
        if cfg.dry_run:
            logger.info('%s cell (%d, %d): ok (dry run)', family, m, n)
            continue
        writeCell(family, m, n, cells, cfg)


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    ids = [data_id for family in cfg.families for data_id in cellIds(family)]
    if cfg.print_commands:
        printCommands(ids, cfg.n_datasets)
        return
    for family in cfg.families:
        generateFamily(family, cfg)
    if not cfg.dry_run:
        printCommands(ids, cfg.n_datasets)


if __name__ == '__main__':
    main()
