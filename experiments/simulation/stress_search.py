"""Stress search: one design value, or the linear-predictor scale, pushed past the training range.

Companion to ood_design.py. Takes the sampled medium test sets, keeps groups, parameters and
priors, and moves one quantity outward in steps, so that NUTS under the stored prior stays
exact and any metabeta-NUTS gap isolates the amortization error. Two axes:

    Leverage point      x_ij = +-k for one random row i and the last continuous predictor j,
        k in {2, 3, 5, 10, 20, 50}, in the standardized units the network sees (the loader
        does not re-standardize stored designs). The sign lowers eta at that row, so a
        Poisson rate never climbs toward the simulator's clip at eta = 10. Only y_i is
        redrawn under the stored parameters (the other rows keep their outcomes, which are
        conditionally independent given theta; Gaussian rows keep their noise realization).
        Stored medium designs have max |x| of 4 (median) to 7.5 (95%), heavy-tailed sources
        up to 25 (99%); Poisson designs are clipped at 5.
    Linear-predictor scale      Bernoulli: ffx and rfx scaled so that sd(eta) = s, s in
        {4, 6, 8, 12, 16}, past the training cap U(2, 4). Gaussian: sigma_eps set so that
        R^2 = r, r in {0.95, 0.97, 0.99, 0.995, 0.999}, past the training cap of 0.95. All
        outcomes are redrawn (Gaussian rows keep their noise realization, rescaled). The
        prior hyperparameters stay as stored, so the truth moves into the prior's tail
        while the prior itself stays inside the trained range (out-of-support priors are
        prior_misspec.py's axis). Poisson has no scale axis: past sd(eta) of about 2 the
        linear predictor exceeds the clip at 10, where the simulator and metabeta target
        exp(min(eta, 10)) but PyMC the exact rate, so the comparison would measure the clip.

Normal outcomes are re-standardized to unit SD exactly as in Simulator.sample (scale
parameters and hyperparameters divided by sd(y)). Datasets come from the fixed permutation
of likelihood_misspec.selectIndices, restricted to datasets with a continuous predictor.
The baseline {size}-{fam}-stressbase holds the selected datasets with their nuts2 fits
sliced from the source (no refits); every other dir gets test.npz and config.yaml only.

Usage (from repo root):
    uv run python experiments/simulation/stress_search.py --families n --n_datasets 4
    uv run python experiments/simulation/stress_search.py --n_datasets 64
    uv run python experiments/simulation/stress_search.py --print_commands
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from likelihood_misspec import (
    FAMILY_IDS,
    FIT_PREFIXES,
    STAT_KEYS,
    _rescaleDataset,
    copyReferenceFit,
    linearPredictorPadded,
)
from metabeta.simulation.simulator import SCALE_PARAMS, simulate
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.families import simulateYNp
from metabeta.utils.preprocessing import checkBinary
from metabeta.utils.logger import setupLogging
from metabeta.utils.experiments import DATA_DIR
from metabeta.utils.fits import REFERENCE_TAG, availableFits, fitPath

logger = logging.getLogger(__name__)

ALL_FAMILIES = ['n', 'b', 'p']
BASE_TAG = 'stressbase'
LEVERAGE_LEVELS = [2, 3, 5, 10, 20, 50]
SCALE_LEVELS: dict[str, list[float]] = {
    'n': [0.95, 0.97, 0.99, 0.995, 0.999],  # R^2
    'b': [4.0, 6.0, 8.0, 12.0, 16.0],  # sd(eta)
    'p': [],
}


def levelTag(axis: str, level: float) -> str:
    if axis == 'leverage':
        return f'lev{int(level)}'
    digits = f'{level:g}'.replace('0.', '').replace('.', '')
    return f'r2{digits}' if level < 1 else f'eta{int(level)}'


def conditions(family: str) -> list[tuple[str, str, float]]:
    """(ds_type tag, axis, level) for one family."""
    out = [(levelTag('leverage', k), 'leverage', float(k)) for k in LEVERAGE_LEVELS]
    out += [(levelTag('scale', s), 'scale', s) for s in SCALE_LEVELS[family]]
    return out


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate stress-search test sets from sampled test data.')
    parser.add_argument('--families', nargs='+', default=ALL_FAMILIES, choices=ALL_FAMILIES, help='likelihood families (letters)')
    parser.add_argument('--size', default='medium', help='size regime of the source sets')
    parser.add_argument('--n_datasets', type=int, default=64, help='datasets per condition')
    parser.add_argument('--seed', type=int, default=0, help='seed for index selection and regeneration')
    parser.add_argument('--overwrite', action='store_true', help='regenerate dirs that already exist')
    parser.add_argument('--print_commands', action='store_true', help='only print the NUTS fit campaign commands and exit')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def continuousColumns(batch: dict[str, np.ndarray], i: int) -> np.ndarray:
    """Indices of dataset ``i``'s non-binary predictor columns (the intercept excluded)."""
    n, d = int(batch['n'][i]), int(batch['d'][i])
    binary = checkBinary(batch['X'][i, :n, 1:d], axis=0)
    return np.flatnonzero(~binary) + 1


def selectIndices(source: dict[str, np.ndarray], n_datasets: int, seed: int) -> np.ndarray:
    """First ``n_datasets`` of likelihood_misspec's permutation that have a continuous predictor."""
    B = source['y'].shape[0]
    order = np.random.default_rng(seed).permutation(B)
    eligible = [j for j in order if continuousColumns(source, j).size > 0]
    if len(eligible) < n_datasets:
        raise ValueError(f'only {len(eligible)} of {B} datasets have a continuous predictor')
    return np.array(eligible[:n_datasets])


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def standardizeOutcome(batch: dict[str, np.ndarray], i: int, y: np.ndarray) -> np.ndarray:
    """Unit-SD outcome with the dataset's scale (hyper)parameters divided accordingly."""
    sd = max(float(y.std()), 1e-6)
    _rescaleDataset(batch, i, sd)
    return y / sd


def leveragePoint(
    batch: dict[str, np.ndarray], i: int, row: int, k: float, lf: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Set x_ij = +-k on row ``row`` and the last continuous column; redraw y_i only."""
    n, q = int(batch['n'][i]), int(batch['q'][i])
    col = int(continuousColumns(batch, i)[-1])
    coef = float(batch['ffx'][i, col])
    if col < q:
        coef += float(batch['rfx'][i, int(batch['groups'][i, row]), col])
    sign = -1 if coef > 0 else 1
    y = batch['y'][i, :n].astype(np.float64)
    if hasSigmaEps(lf):
        eps = y - linearPredictorPadded(batch, i)
    batch['X'][i, row, col] = sign * k
    eta = linearPredictorPadded(batch, i)
    if hasSigmaEps(lf):
        y = eta + eps
        y = standardizeOutcome(batch, i, y)
    else:
        y[row] = simulateYNp(rng, eta[row : row + 1], 0.0, lf)[0]
    batch['y'][i, :n] = y.astype(batch['y'].dtype)
    return col, sign


def scaleLinearPredictor(
    batch: dict[str, np.ndarray], i: int, level: float, lf: int, rng: np.random.Generator
) -> float:
    """Bernoulli: sd(eta) = level; Gaussian: R^2 = level. Returns the realized value."""
    n, d, q = int(batch['n'][i]), int(batch['d'][i]), int(batch['q'][i])
    eta = linearPredictorPadded(batch, i)
    if hasSigmaEps(lf):
        signal_var = float(eta.var())
        sigma_old = float(batch['sigma_eps'][i])
        sigma_new = float(np.sqrt(signal_var * (1.0 - level) / level))
        eps = (batch['y'][i, :n].astype(np.float64) - eta) * (sigma_new / sigma_old)
        batch['sigma_eps'][i] = sigma_new
        y = standardizeOutcome(batch, i, eta + eps)
        realized = 1.0 - float(batch['sigma_eps'][i]) ** 2
    else:
        scale = level / float(eta.std())
        for key in SCALE_PARAMS - {'sigma_eps'}:
            batch[key][i] = batch[key][i] * scale
        params = {'ffx': batch['ffx'][i, :d], 'rfx': batch['rfx'][i, :, :q]}
        obs = {'X': batch['X'][i, :n, :d], 'groups': batch['groups'][i, :n].astype(np.int64)}
        y = simulate(rng, params, obs, lf)
        realized = float(linearPredictorPadded(batch, i).std())
    batch['y'][i, :n] = y.astype(batch['y'].dtype)
    return realized


def regenerate(
    batch: dict[str, np.ndarray],
    family: str,
    axis: str,
    level: float,
    rows: list[int],
    rng: np.random.Generator,
) -> dict[str, list]:
    """Apply the condition to every dataset in place; returns the per-dataset records.

    ``rows`` are the leverage rows, drawn once per family so that the levels of one dataset
    move the same value outward.
    """
    lf = FAMILY_IDS[family]
    B = batch['y'].shape[0]
    records: dict[str, list] = {'row': [], 'col': [], 'sign': []} if axis == 'leverage' else {'realized': []}
    for i in range(B):
        if axis == 'leverage':
            col, sign = leveragePoint(batch, i, rows[i], level, lf, rng)
            records['row'].append(rows[i])
            records['col'].append(col)
            records['sign'].append(sign)
        else:
            records['realized'].append(scaleLinearPredictor(batch, i, level, lf, rng))
    return records


# ---------------------------------------------------------------------------
# Directory writing
# ---------------------------------------------------------------------------


def writeConfig(
    src_dir: Path,
    out_dir: Path,
    data_id: str,
    idx: np.ndarray,
    seed: int,
    axis: str,
    level: float,
    records: dict[str, list],
) -> None:
    with open(src_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['data_id'] = data_id
    cfg['bs_test'] = len(idx)
    cfg['stress_axis'] = axis
    cfg['stress_level'] = level
    cfg['stress_source'] = src_dir.name
    cfg['stress_orig_indices'] = [int(j) for j in idx]
    cfg['stress_seed'] = seed
    for key, values in records.items():
        cfg[f'stress_{key}'] = [float(v) if isinstance(v, float) else int(v) for v in values]
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def clearStaleFits(out_dir: Path, data_id: str) -> None:
    """Regenerated data invalidate every fit of the previous data in ``out_dir``."""
    stale = [fitPath(out_dir / 'test.npz', t) for t in availableFits(out_dir / 'test.npz')]
    stale += sorted((out_dir / 'fits').glob('*.npz'))
    for p in stale:
        p.unlink()
    if stale:
        logger.warning('%s: removed %d stale fit file(s)', data_id, len(stale))


def generateFamily(cfg: argparse.Namespace, family: str) -> list[str]:
    """Write the baseline and condition dirs for one family; returns the data ids."""
    src_id = f'{cfg.size}-{family}-sampled'
    src_dir = DATA_DIR / src_id
    test_path = src_dir / 'test.npz'
    if not test_path.exists() or not fitPath(test_path, REFERENCE_TAG).exists():
        logger.warning('%s: test.npz or its %s fit missing - skipping', src_id, REFERENCE_TAG)
        return []
    with np.load(test_path, allow_pickle=True) as z:
        source = {k: z[k] for k in z.files}
    idx = selectIndices(source, cfg.n_datasets, cfg.seed)
    # 200-offset on the family token keeps streams disjoint from the misspec scripts
    rng_rows = np.random.default_rng([cfg.seed, 200 + FAMILY_IDS[family]])
    rows = [int(rng_rows.integers(int(source['n'][j]))) for j in idx]

    base_id = f'{cfg.size}-{family}-{BASE_TAG}'
    base_dir = DATA_DIR / base_id
    created = [base_id]
    if base_dir.exists() and not cfg.overwrite:
        logger.info('%s: exists - skipping (use --overwrite)', base_id)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base_dir / 'test.npz', **{k: v[idx] for k, v in source.items()})
        copyReferenceFit(test_path, base_dir / 'test.npz', idx)
        writeConfig(src_dir, base_dir, base_id, idx, cfg.seed, 'none', 0.0, {})
        logger.info('%s: baseline written (%d datasets)', base_id, len(idx))

    for cond_pos, (tag, axis, level) in enumerate(conditions(family)):
        data_id = f'{cfg.size}-{family}-{tag}'
        out_dir = DATA_DIR / data_id
        created.append(data_id)
        if out_dir.exists() and not cfg.overwrite:
            logger.info('%s: exists - skipping (use --overwrite)', data_id)
            continue
        if out_dir.exists():
            clearStaleFits(out_dir, data_id)
        batch = {
            k: v[idx].copy()
            for k, v in source.items()
            if k not in STAT_KEYS and not k.startswith(FIT_PREFIXES)
        }
        rng = np.random.default_rng([cfg.seed, 200 + FAMILY_IDS[family], cond_pos])
        records = regenerate(batch, family, axis, level, rows, rng)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / 'test.npz', **batch)
        writeConfig(src_dir, out_dir, data_id, idx, cfg.seed, axis, level, records)
        logger.info('%s: written (%s %g, %d datasets)', data_id, axis, level, len(idx))
    return created


# ---------------------------------------------------------------------------
# Fit campaign
# ---------------------------------------------------------------------------


def printCommands(families: list[str], size: str, n_datasets: int) -> None:
    for family in families:
        for tag, _, _ in conditions(family):
            print(
                f'scripts/fit-ref.sh --method nuts --level 2 '
                f'--n_datasets {n_datasets} --data_id {size}-{family}-{tag}'
            )
    print('# after the fits finished (checks the fits, then reintegrates):')
    for family in families:
        for tag, _, _ in conditions(family):
            print(
                f'uv run python metabeta/simulation/check.py --partition test '
                f'--data_id {size}-{family}-{tag}'
            )


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    if cfg.print_commands:
        printCommands(cfg.families, cfg.size, cfg.n_datasets)
        return
    created = []
    for family in cfg.families:
        created += generateFamily(cfg, family)
    if not created:
        logger.error('No data dirs generated.')
        return
    print(f'\nGenerated/verified {len(created)} data dirs:')
    for data_id in created:
        print(f'  {data_id}')
    print('\nNUTS fit campaign (the baseline dirs carry their fits):')
    printCommands(cfg.families, cfg.size, cfg.n_datasets)


if __name__ == '__main__':
    main()
