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

--extend adds the cells (600, 3000) and (600, 9000) without touching the sets above. It
rebuilds the masters, checks them against the written sets, and adds 200 groups of 15 rows to
each: new design rows from the master's own column distributions and correlation map, put
through the master's predictor transform (its moments, not recomputed, so the existing groups
keep their X and the parameters stay the truth), new random effects from the master's
sigma_rfx, and outcomes from its final parameters, calibration and Gaussian scaling included
and not redone. The (200, 9000) cell would need 45 rows in every group and is left out.

Usage from repo root:
    uv run python experiments/simulation/group_budget_sets.py --families n --n_datasets 2 --dry_run
    uv run python experiments/simulation/group_budget_sets.py
    uv run python experiments/simulation/group_budget_sets.py --print_commands
    uv run python experiments/simulation/group_budget_sets.py --extend
"""

import argparse
import logging

import numpy as np
import yaml

from metabeta.simulation import Prior, Simulator, Synthesizer, hypersample
from metabeta.simulation.generate import Generator
from metabeta.simulation.simulator import linearPredictor, simulate
from metabeta.simulation.synthesizer import DISTS, PROBS
from metabeta.utils.constants import FAMILY_NAMES_REVERSE
from metabeta.utils.experiments import DATA_DIR
from metabeta.utils.families import POISSON_ETA_CLIP_MAX, POISSON_X_CLIP_ABS
from metabeta.utils.logger import setupLogging
from metabeta.utils.padding import aggregate
from metabeta.utils.preprocessing import checkBinary, checkContinuous, checkCountLike, moments
from metabeta.utils.sampling import wishartCorrelation
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
# three times the limits, from each master extended by EXT_M groups of EXT_NI rows
EXT_CELLS = [(600, 3000), (600, 9000)]
EXT_M, EXT_NI = 200, 15
EXT_ENTROPY = 3  # extra seed entropy of the extension, so its draws are independent of the master's
MAX_REDRAWS = 50


def cellTag(m: int, n: int) -> str:
    return f'gb{m}x{n}'


def cellIds(family: str, cells: list[tuple[int, int]] = CELLS) -> list[str]:
    return [f'{SIZE}-{family}-{cellTag(m, n)}' for m, n in cells]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate the nested group-budget test sets.')
    parser.add_argument('--families', nargs='+', default=['n', 'b', 'p'], choices=['n', 'b', 'p'])
    parser.add_argument('--n_datasets', type=int, default=16, help='master datasets per family')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true', help='generate and check, write nothing')
    parser.add_argument('--print_commands', action='store_true', help='only print the fit campaign')
    parser.add_argument('--extend', action='store_true', help='the 3x cells from extended masters')
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def cellNs(m: int, n: int) -> np.ndarray:
    ns = np.full(m, n // m)
    ns[: n % m] += 1
    return ns


def cellRows(ns: np.ndarray, master_ns: np.ndarray) -> np.ndarray:
    """Row indices of a master with group sizes master_ns taking the first ns[g] rows of group g."""
    start = np.concatenate([[0], np.cumsum(master_ns)[:-1]])
    return np.concatenate([start[g] + np.arange(k) for g, k in enumerate(ns)])


def cell(master: dict[str, np.ndarray], m: int, n: int) -> dict[str, np.ndarray]:
    ns = cellNs(m, n)
    assert np.all(ns <= master['ns'][:m])
    rows = cellRows(ns, master['ns'])
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


class ExtensibleSynthesizer(Synthesizer):
    """Synthesizer that can draw more rows from the design it sampled.

    _sample and _induceCorrelation repeat Synthesizer's (metabeta/simulation/synthesizer.py)
    draw for draw, so a master comes out unchanged; they also keep the sampled design x, the
    column distributions and the correlation map, an affine map fixed by the sample's moments.
    `rows` puts new draws of the distributions through that map, so they follow the same design.
    """

    def _sample(self, n: int, d: int) -> np.ndarray:
        assert d > 1 and self.correlate and not self.toy
        kinds = self.rng.choice(DISTS, size=d - 1, replace=True, p=PROBS)
        self.dists, columns = [], []
        for kind in kinds:
            self.dists.append(kind(self.rng))
            columns.append(self.dists[-1].sample(n))
        self.x = np.ones((n, d))
        self.x[:, 1:] = self._induceCorrelation(np.column_stack(columns))
        return self.x

    def _induceCorrelation(self, x: np.ndarray) -> np.ndarray:
        self.cont = checkContinuous(x)
        d_ = self.cont.sum()
        if d_ < 2:
            self.map = None
            return x
        L = np.linalg.cholesky(wishartCorrelation(self.rng, d_, nu=d_ + 9))
        x_ = x[:, self.cont]
        mean = np.mean(x_, axis=0, keepdims=True)
        std = np.std(x_, axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        std_cor = np.std(((x_ - mean) / std) @ L.T, axis=0, keepdims=True)
        std_cor = np.where(std_cor < 1e-12, 1.0, std_cor)
        self.map = (mean, std, L, std_cor)
        out = x.copy()
        out[:, self.cont] = self._correlate(x_)
        return out

    def _correlate(self, x_: np.ndarray) -> np.ndarray:
        mean, std, L, std_cor = self.map
        return ((x_ - mean) / std) @ L.T / std_cor * std + mean

    def rows(self, rng: np.random.Generator, k: int) -> np.ndarray:
        """k new design rows, intercept first, drawn with rng."""
        for dist in self.dists:
            dist.rng = rng
        x = np.column_stack([dist.sample(k) for dist in self.dists]).astype(float)
        if self.map is not None:
            x[:, self.cont] = self._correlate(x[:, self.cont])
        return np.concatenate([np.ones((k, 1)), x], axis=1)


def transformLike(x: np.ndarray, x_master: np.ndarray, lf: int) -> np.ndarray:
    """Simulator.sample's predictor transform of x_master (transformPredictors, then the Poisson
    clip), applied to x with x_master's count columns and moments."""
    counts = checkCountLike(x_master)
    t_master, t = x_master.astype(float, copy=True), x.astype(float, copy=True)
    t_master[:, counts] = np.log1p(t_master[:, counts])
    t[:, counts] = np.log1p(t[:, counts])
    mean, std = moments(t_master, 0, exclude=checkBinary(t_master))
    std = np.where((~np.isfinite(std)) | (std < 1e-6), 1.0, std)
    out = (t - mean) / std
    if lf == 2:
        out[:, 1:] = np.clip(out[:, 1:], -POISSON_X_CLIP_ABS, POISSON_X_CLIP_ABS)
    return out


def masterProblem(
    master: dict[str, np.ndarray], lf: int, cells: list[tuple[int, int]] = CELLS
) -> str | None:
    """Why a master must be redrawn, or None."""
    for m, n in cells:
        x = cell(master, m, n)['X'][:, 1:]
        if np.any(x.std(0) == 0):
            return f'constant predictor in cell ({m}, {n})'
    if lf == 2:
        eta = linearPredictor(master, master)
        if eta.max() > POISSON_ETA_CLIP_MAX:
            return f'true eta {eta.max():.2f} exceeds the Poisson clip'
    return None


def drawMaster(
    seedseq: np.random.SeedSequence, lf: int
) -> tuple[dict[str, np.ndarray], ExtensibleSynthesizer, int]:
    """One master dataset, its design and the number of redraws it took."""
    for attempt, seed in enumerate(seedseq.spawn(MAX_REDRAWS)):
        rng = np.random.default_rng(seed)
        prior = Prior(rng, hypersample(rng, D, Q, likelihood_family=lf))
        ns = np.full(MASTER_M, MASTER_NI)
        design = ExtensibleSynthesizer(rng)
        master = Simulator(rng, prior, design, ns).sample()
        assert np.array_equal(master['groups'], np.repeat(np.arange(MASTER_M), MASTER_NI))
        problem = masterProblem(master, lf)
        if problem is None:
            return master, design, attempt
        logger.info('  redraw (%s)', problem)
    raise RuntimeError(f'no valid master after {MAX_REDRAWS} draws')


def extendMaster(
    master: dict[str, np.ndarray],
    design: ExtensibleSynthesizer,
    seedseq: np.random.SeedSequence,
    lf: int,
) -> tuple[dict[str, np.ndarray], int]:
    """The master with EXT_M more groups of EXT_NI rows and the number of redraws it took."""
    assert np.array_equal(transformLike(design.x, design.x, lf), master['X'])
    assert master['rfx'].shape == (MASTER_M, 1) and float(master['eta_rfx']) == 0
    groups = MASTER_M + np.repeat(np.arange(EXT_M), EXT_NI)
    for attempt, seed in enumerate(seedseq.spawn(MAX_REDRAWS)):
        rng = np.random.default_rng(seed)
        x = transformLike(design.rows(rng, EXT_M * EXT_NI), design.x, lf)
        rfx = master['sigma_rfx'] * rng.standard_normal((EXT_M, 1))
        ext = dict(master)
        ext.update(
            X=np.concatenate([master['X'], x]),
            groups=np.concatenate([master['groups'], groups]),
            rfx=np.concatenate([master['rfx'], rfx]),
            ns=np.concatenate([master['ns'], np.full(EXT_M, EXT_NI)]),
            m=np.array(MASTER_M + EXT_M),
            n=np.array(len(master['y']) + len(groups)),
        )
        y = simulate(rng, ext, {'X': x, 'groups': groups}, lf)
        ext['y'] = np.concatenate([master['y'], y])
        problem = masterProblem(ext, lf, EXT_CELLS)
        if problem is None:
            return ext, attempt
        logger.info('  redraw extension (%s)', problem)
    raise RuntimeError(f'no valid extension after {MAX_REDRAWS} draws')


def checkWritten(data_id: str, cells: list[dict]) -> None:
    """The rebuilt cells equal the written test set on every array this script writes."""
    batch = Generator._castCompactTypes(aggregate(cells))
    with np.load(DATA_DIR / data_id / 'test.npz', allow_pickle=True) as z:
        for k, v in batch.items():
            if not np.array_equal(z[k], v, equal_nan=v.dtype.kind == 'f'):
                raise AssertionError(f'{data_id}: rebuilt {k} differs from the written set')


def writeCell(
    family: str, m: int, n: int, cells: list[dict], cfg: argparse.Namespace, **meta
) -> None:
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
        **meta,
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
    masters, designs = [], []
    for i, seedseq in enumerate(root.spawn(cfg.n_datasets)):
        master, design, redraws = drawMaster(seedseq, lf)
        logger.info('%s master %d: %d redraw(s)', family, i, redraws)
        masters.append(master)
        designs.append(design)
    if cfg.extend:
        extendFamily(family, lf, masters, designs, cfg)
        return
    for m, n in CELLS:
        cells = [cell(master, m, n) for master in masters]
        if cfg.dry_run:
            logger.info('%s cell (%d, %d): ok (dry run)', family, m, n)
            continue
        writeCell(family, m, n, cells, cfg, group_budget_master=[MASTER_M, MASTER_NI])


def extendFamily(
    family: str,
    lf: int,
    masters: list[dict],
    designs: list[ExtensibleSynthesizer],
    cfg: argparse.Namespace,
) -> None:
    for data_id, (m, n) in zip(cellIds(family), CELLS):
        checkWritten(data_id, [cell(master, m, n) for master in masters])
    logger.info('%s: rebuilt masters reproduce the written sets', family)
    root = np.random.SeedSequence([cfg.seed, lf, EXT_ENTROPY])
    extended = []
    for i, seedseq in enumerate(root.spawn(cfg.n_datasets)):
        ext, redraws = extendMaster(masters[i], designs[i], seedseq, lf)
        logger.info('%s extension %d: %d redraw(s)', family, i, redraws)
        extended.append(ext)
    for m, n in EXT_CELLS:
        cells = [cell(ext, m, n) for ext in extended]
        if cfg.dry_run:
            logger.info('%s cell (%d, %d): ok (dry run)', family, m, n)
            continue
        writeCell(
            family,
            m,
            n,
            cells,
            cfg,
            group_budget_master=[[MASTER_M, MASTER_NI], [EXT_M, EXT_NI]],
            group_budget_extension_entropy=[cfg.seed, lf, EXT_ENTROPY],
        )


def printCommands(ids: list[str], n_datasets: int) -> None:
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
