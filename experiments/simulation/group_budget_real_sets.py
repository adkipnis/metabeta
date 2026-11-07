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

--extend adds the cells (600, 3000) and (600, 9000) of group_budget_sets.py without touching the
sets above. It rebuilds the masters, checks them against the written sets, and adds to each 200
groups of 15 shuffled rows, drawn at random from the source groups with >= 15 rows that the
master does not hold. Predictors and Gaussian outcomes are standardized anew on the extended
master; the prior stays the master's.

Usage from repo root:
    python experiments/simulation/group_budget_real_sets.py --dry_run
    python experiments/simulation/group_budget_real_sets.py
    python experiments/simulation/group_budget_real_sets.py --print_commands
    python experiments/simulation/group_budget_real_sets.py --extend
"""

import argparse
import logging

import numpy as np
import yaml
from group_budget_sets import (
    CELLS,
    EXT_CELLS,
    EXT_ENTROPY,
    EXT_M,
    EXT_NI,
    SIZE,
    D,
    Q,
    cellNs,
    cellRows,
    checkWritten,
    printCommands,
)

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


def cellIds(family: str, cells: list[tuple[int, int]] = CELLS) -> list[str]:
    return [f'{SIZE}-{family}-{cellTag(m, n)}' for m, n in cells]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate the nested real-data group-budget sets.')
    parser.add_argument('--families', nargs='+', default=['n', 'b', 'p'], choices=['n', 'b', 'p'])
    parser.add_argument('--n_datasets', type=int, default=16, help='masters per family')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true', help='generate and check, write nothing')
    parser.add_argument('--print_commands', action='store_true', help='only print the fit campaign')
    parser.add_argument('--extend', action='store_true', help='the 3x cells from extended masters')
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def loadSource(family: str) -> dict[str, np.ndarray]:
    src = dict(np.load(SOURCES[family], allow_pickle=True))
    assert np.all(np.diff(src['groups']) >= 0), 'source groups must be sorted'
    assert np.array_equal(np.unique(src['groups']), np.arange(len(src['ns'])))
    src['start'] = np.concatenate([[0], np.cumsum(src['ns'])[:-1]])
    src['name'] = SOURCES[family].stem
    return src


def cell(master: dict[str, np.ndarray], m: int, n: int) -> dict[str, np.ndarray]:
    ns = cellNs(m, n)
    assert np.all(ns <= master['ns'][:m])
    rows = cellRows(ns, master['ns'])
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


def constantCells(
    x: np.ndarray, ns: np.ndarray, cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The cells in which a column of x, a master's raw predictors, is constant."""
    return [(m, n) for m, n in cells if np.any(x[cellRows(cellNs(m, n), ns)].std(0) == 0)]


def assemble(
    src: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray, ns: np.ndarray, lf: int
) -> dict[str, np.ndarray]:
    """The data of a master from its source rows and columns, standardized on those rows."""
    x = transformPredictors(
        src['X'][rows][:, cols], axis=0, exclude_binary=True, transform_counts=True
    )
    x = np.concatenate([np.ones((len(x), 1)), x], axis=-1)
    y = src['y'][rows]
    sd_y = max(float(y.std()), 1e-6) if lf == 0 else 1.0
    if lf == 0:
        y = y / sd_y
    return {
        'y': y,
        'X': x,
        'groups': np.repeat(np.arange(len(ns)), ns),
        'm': np.array(len(ns)),
        'n': np.array(len(rows)),
        'ns': ns,
        'd': np.array(D),
        'q': np.array(Q),
        'sd_y': np.array(sd_y),
        'source': np.array(src['name'], dtype=object),
        'source_rows': rows,
        'source_cols': cols,
    }


def drawMaster(
    seedseq: np.random.SeedSequence, src: dict[str, np.ndarray], lf: int
) -> tuple[dict[str, np.ndarray], int]:
    """One master dataset and the number of redraws it took."""
    for attempt, seed in enumerate(seedseq.spawn(MAX_REDRAWS)):
        rng = np.random.default_rng(seed)
        n_cols = src['X'].shape[1]
        cols = np.sort(rng.choice(n_cols, D - 1, replace=False))
        rows = sourceRows(rng, src)
        constant = constantCells(src['X'][rows][:, cols], MASTER_NS, CELLS)
        if constant:
            logger.info('  redraw (constant predictor in cells %s)', constant)
            continue
        master = {
            'ffx': np.full(D, np.nan),
            'sigma_rfx': np.full(Q, np.nan),
            'corr_rfx': np.full((Q, Q), np.nan),
            'rfx': np.full((len(MASTER_NS), Q), np.nan),
            **{k: np.asarray(v) for k, v in hypersample(rng, D, Q, likelihood_family=lf).items()},
            **assemble(src, rows, cols, MASTER_NS, lf),
        }
        if hasSigmaEps(lf):
            master['sigma_eps'] = np.array(np.nan)
            master['r_squared'] = np.array(np.nan)
        return master, attempt
    raise RuntimeError(f'no valid master after {MAX_REDRAWS} draws')


def extendMaster(
    master: dict[str, np.ndarray],
    seedseq: np.random.SeedSequence,
    src: dict[str, np.ndarray],
    lf: int,
) -> tuple[dict[str, np.ndarray], int]:
    """The master with EXT_M more source groups of EXT_NI rows and the number of redraws it took."""
    held = np.unique(src['groups'][master['source_rows']])
    pool = np.setdiff1d(np.flatnonzero(src['ns'] >= EXT_NI), held)
    cols = master['source_cols']
    ns = np.concatenate([master['ns'], np.full(EXT_M, EXT_NI)])
    for attempt, seed in enumerate(seedseq.spawn(MAX_REDRAWS)):
        rng = np.random.default_rng(seed)
        members = rng.choice(pool, EXT_M, replace=False)
        new = [src['start'][g] + rng.permutation(src['ns'][g])[:EXT_NI] for g in members]
        rows = np.concatenate([master['source_rows'], *new])
        constant = constantCells(src['X'][rows][:, cols], ns, EXT_CELLS)
        if constant:
            logger.info('  redraw extension (constant predictor in cells %s)', constant)
            continue
        ext = {**master, **assemble(src, rows, cols, ns, lf)}
        ext['rfx'] = np.full((len(ns), Q), np.nan)
        return ext, attempt
    raise RuntimeError(f'no valid extension after {MAX_REDRAWS} draws')


def checkLabels(cells: list[dict[str, np.ndarray]], src: dict[str, np.ndarray]) -> None:
    """Every label of a cell covers exactly one source group, and no source group twice."""
    for c in cells:
        pairs = np.unique(np.stack([c['groups'], src['groups'][c['source_rows']]]), axis=1)
        assert pairs.shape[1] == int(c['m']), 'a label spans several source groups'
        assert len(np.unique(pairs[1])) == int(c['m']), 'a source group carries several labels'
        assert np.array_equal(np.bincount(c['groups']), c['ns'])


def writeCell(
    family: str, m: int, n: int, cells: list[dict], cfg: argparse.Namespace, **meta
) -> None:
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
        **meta,
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
    if cfg.extend:
        extendFamily(family, lf, src, masters, cfg)
        return
    for m, n in CELLS:
        cells = [cell(master, m, n) for master in masters]
        checkLabels(cells, src)
        if cfg.dry_run:
            logger.info('%s cell (%d, %d): ok (dry run)', family, m, n)
            continue
        writeCell(family, m, n, cells, cfg, group_budget_master_ns=[[200, 30], [200, 15]])


def extendFamily(
    family: str, lf: int, src: dict[str, np.ndarray], masters: list[dict], cfg: argparse.Namespace
) -> None:
    for data_id, (m, n) in zip(cellIds(family), CELLS):
        cells = [cell(master, m, n) for master in masters]
        side = {k: np.stack([c.pop(k) for c in cells]) for k in ('source_rows', 'source_cols')}
        checkWritten(data_id, cells)
        with np.load(DATA_DIR / data_id / 'source_rows.npz', allow_pickle=True) as z:
            assert all(np.array_equal(z[k], v) for k, v in side.items()), data_id
    logger.info('%s: rebuilt masters reproduce the written sets', family)
    root = np.random.SeedSequence([cfg.seed, lf, EXT_ENTROPY])
    extended = []
    for i, seedseq in enumerate(root.spawn(cfg.n_datasets)):
        ext, redraws = extendMaster(masters[i], seedseq, src, lf)
        logger.info('%s extension %d: %d redraw(s)', family, i, redraws)
        extended.append(ext)
    for m, n in EXT_CELLS:
        cells = [cell(ext, m, n) for ext in extended]
        checkLabels(cells, src)
        if cfg.dry_run:
            logger.info('%s cell (%d, %d): ok (dry run)', family, m, n)
            continue
        writeCell(
            family,
            m,
            n,
            cells,
            cfg,
            group_budget_master_ns=[[200, 30], [400, 15]],
            group_budget_extension_entropy=[cfg.seed, lf, EXT_ENTROPY],
        )


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    cells = EXT_CELLS if cfg.extend else CELLS
    ids = [data_id for family in cfg.families for data_id in cellIds(family, cells)]
    if cfg.print_commands:
        printCommands(ids, cfg.n_datasets)
        return
    for family in cfg.families:
        generateFamily(family, cfg)
    if not cfg.dry_run:
        printCommands(ids, cfg.n_datasets)


if __name__ == '__main__':
    main()
