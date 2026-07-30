"""Predictor misspecification: regenerate test designs from out-of-distribution sources.

Companion to likelihood_misspec.py, moving the contamination from the conditional outcome
distribution to the design distribution.  Takes existing sampled test datasets, keeps the
grouping structure and all generating parameters, replaces the predictors X wholesale with
draws from sources outside the training simulator, and regenerates y under the ORIGINAL
likelihood.  The fitted model therefore stays correctly specified — y | X, θ follows the
stored family and priors exactly — so NUTS remains an exact reference and any metabeta↔NUTS
divergence isolates the amortization gap on out-of-distribution designs.

Conditions (family-independent, with a severity dial where applicable):

    Heavy-tailed marginals   X_ij ~ t_ν i.i.d., ν ∈ {2, 1}.  Training designs have finite
        variance by construction (scamd draws Student-t causes with ν ≥ 2.5, clamps them
        to ±10 and standardizes; all other cause families are lighter-tailed), so ν ≤ 2
        (infinite variance; ν = 1 is Cauchy) lies outside the training support.
    Tail dependence          Clayton copula over the predictors with standard-normal
        marginals, Kendall τ ∈ {0.5, 0.9}.  The training copula is Gaussian, which has
        zero tail dependence at any correlation strength; Clayton has λ_L = 2^{-1/θ} > 0.
    Longitudinal design      the first predictor is a centered within-group time index
        (growth-curve design, deterministic and repeated across groups); remaining
        predictors are i.i.d. standard normal.  Deterministic repeated designs cannot
        arise from the pooled-row SCM or tabular training sources.

Replacement designs pass through the same transformPredictors() standardization as the
simulator (plus the Poisson X clip), and normal outcomes are re-standardized to unit SD
exactly as in Simulator.sample, keeping parameters and prior context coherent.  Intercept-
only datasets (d = 1) have no predictors to replace and only receive fresh outcome noise.

Note: for Bernoulli/Poisson sources the stored parameters were LP-scale-calibrated to the
original design, so heavy-tailed replacements can push the linear predictor into clipped or
saturated regimes; start with --families n (default) and treat b/p runs as exploratory.

The severity-0 baseline is the shared {size}-{fam}-misbase dir from likelihood_misspec.py
(same unsorted index prefix, existing NUTS fits reused — zero refits); it is created here
when missing.  Each condition produces a standard-format dir {size}-{fam}-{tag} with
test.npz only — NUTS fits to be produced with fit.py.

Usage (from repo root):
    uv run python experiments/simulation/ood_design.py --families n --sizes small --n_datasets 4
    uv run python experiments/simulation/ood_design.py --n_datasets 32
    uv run python experiments/simulation/ood_design.py --print_commands

After generation, produce NUTS fits per condition dir (cluster):
    sbatch --array=0-31 scripts/fit-nuts.sh --data_id small-n-xt2
then reintegrate (from metabeta/simulation/):
    uv run python fit.py --size small --family 0 --ds_type xt2 --reintegrate
and optionally precompute analytical stats (from metabeta/analytical/):
    uv run python precompute.py --size small --family 0 --ds_type xt2 --partition test
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.special import ndtri

from likelihood_misspec import (
    BASE_TAG,
    CONDITIONS as LIK_CONDITIONS,
    DEFAULT_SIZES,
    FAMILY_IDS,
    FIT_PREFIXES,
    STAT_KEYS,
    _rescaleDataset,
    selectIndices,
    sliceNpzStreaming,
)
from metabeta.simulation.simulator import simulate
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.families import POISSON_X_CLIP_ABS
from metabeta.utils.preprocessing import transformPredictors
from metabeta.utils.logger import setupLogging
from metabeta.utils.experiments import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_FAMILIES = ['n']
ALL_FAMILIES = ['n', 'b', 'p']

# (ds_type tag, kind, severity); severity is df / Kendall τ / unused.
CONDITIONS: list[tuple[str, str, float]] = [
    ('xt2', 'xtail', 2.0),
    ('xt1', 'xtail', 1.0),
    ('clay5', 'clayton', 0.5),
    ('clay9', 'clayton', 0.9),
    ('xlong', 'longitudinal', 0.0),
]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate predictor-misspecified test sets from sampled test data.')
    parser.add_argument('--families', nargs='+', default=DEFAULT_FAMILIES, choices=ALL_FAMILIES, help='likelihood families (letters); b/p are exploratory')
    parser.add_argument('--sizes', nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES, help='size regimes')
    parser.add_argument('--n_datasets', type=int, default=32, help='datasets per condition (subset of the 512 test datasets)')
    parser.add_argument('--seed', type=int, default=0, help='seed for index selection and X/y regeneration (must match likelihood_misspec.py for baseline reuse)')
    parser.add_argument('--overwrite', action='store_true', help='regenerate dirs that already exist')
    parser.add_argument('--print_commands', action='store_true', help='only print the NUTS fit campaign commands and exit')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Design construction
# ---------------------------------------------------------------------------


def sampleClaytonUniform(rng: np.random.Generator, n: int, k: int, tau: float) -> np.ndarray:
    """Clayton copula uniforms via the Marshall–Olkin frailty construction; τ = θ/(θ+2)."""
    theta = 2.0 * tau / (1.0 - tau)
    v = np.maximum(rng.gamma(1.0 / theta, 1.0, size=n), 1e-300)
    e = rng.exponential(size=(n, k))
    return (1.0 + e / v[:, None]) ** (-1.0 / theta)


def buildDesign(
    rng: np.random.Generator,
    kind: str,
    severity: float,
    n: int,
    d: int,
    groups: np.ndarray,
) -> np.ndarray:
    """Raw (pre-standardization) replacement design with intercept column."""
    X = np.ones((n, d))
    if d == 1:
        return X
    k = d - 1
    if kind == 'xtail':
        X[:, 1:] = rng.standard_t(severity, size=(n, k))
    elif kind == 'clayton':
        u = sampleClaytonUniform(rng, n, k, severity)
        X[:, 1:] = ndtri(np.clip(u, 1e-12, 1.0 - 1e-12))
    elif kind == 'longitudinal':
        t = np.empty(n)
        for g in np.unique(groups):
            sel = groups == g
            c = int(sel.sum())
            t[sel] = np.arange(c) - (c - 1) / 2.0  # centered: stays continuous downstream
        X[:, 1] = t
        if k > 1:
            X[:, 2:] = rng.standard_normal((n, k - 1))
    else:
        raise ValueError(f'unknown misspecification kind: {kind}')
    return X


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def regenerateXY(
    batch: dict[str, np.ndarray],
    family: str,
    kind: str,
    severity: float,
    rng: np.random.Generator,
) -> None:
    """Replace every dataset's design in place and redraw y under the original likelihood."""
    lf = FAMILY_IDS[family]
    B = batch['y'].shape[0]
    n_intercept_only = 0
    for i in range(B):
        n = int(batch['n'][i])
        d = int(batch['d'][i])
        q = int(batch['q'][i])
        if d == 1:
            n_intercept_only += 1
        groups = batch['groups'][i, :n].astype(np.int64)
        X = buildDesign(rng, kind, severity, n, d, groups)
        X = transformPredictors(X, axis=0, exclude_binary=True, transform_counts=True)
        if lf == 2 and d > 1:
            X[:, 1:] = np.clip(X[:, 1:], -POISSON_X_CLIP_ABS, POISSON_X_CLIP_ABS)
        params = {
            'ffx': batch['ffx'][i, :d].astype(np.float64),
            'rfx': batch['rfx'][i, :, :q].astype(np.float64),
        }
        if 'sigma_eps' in batch:
            params['sigma_eps'] = float(batch['sigma_eps'][i])
        y = simulate(rng, params, {'X': X, 'groups': groups}, lf)
        if hasSigmaEps(lf):
            sd = max(float(y.std()), 1e-6)
            y = y / sd
            _rescaleDataset(batch, i, sd)
        batch['X'][i] = 0.0
        batch['X'][i, :n, :d] = X.astype(batch['X'].dtype)
        batch['y'][i, :n] = y.astype(batch['y'].dtype)
        batch['y'][i, n:] = 0.0
    if n_intercept_only:
        logger.info(
            '%d/%d datasets are intercept-only (d=1): design unchanged, outcomes redrawn',
            n_intercept_only,
            B,
        )


# ---------------------------------------------------------------------------
# Directory writing
# ---------------------------------------------------------------------------


def writeConfig(
    src_dir: Path,
    out_dir: Path,
    data_id: str,
    n_datasets: int,
    kind: str,
    severity: float,
    idx: np.ndarray,
    seed: int,
    target: str | None = None,
) -> None:
    with open(src_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['data_id'] = data_id
    cfg['bs_test'] = n_datasets
    cfg['misspec_kind'] = kind
    cfg['misspec_severity'] = severity
    if target is not None:
        cfg['misspec_target'] = target
    cfg['misspec_source'] = src_dir.name
    cfg['misspec_orig_indices'] = [int(j) for j in idx]
    cfg['misspec_seed'] = seed
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensureBaseline(
    cfg: argparse.Namespace,
    family: str,
    size: str,
    src_dir: Path,
    source: dict[str, np.ndarray],
    idx: np.ndarray,
) -> str:
    """Reuse the shared likelihood_misspec baseline dir, creating or extending it if needed.

    Valid iff the stored index list has our idx as a prefix (selectIndices returns prefixes
    of a fixed permutation, so a larger existing baseline remains valid for smaller runs).
    """
    base_id = f'{size}-{family}-{BASE_TAG}'
    base_dir = DATA_DIR / base_id
    ours = [int(j) for j in idx]

    if base_dir.exists() and not cfg.overwrite:
        with open(base_dir / 'config.yaml') as f:
            base_cfg = yaml.safe_load(f)
        stored = base_cfg.get('misspec_orig_indices', [])
        if len(stored) >= len(ours) and stored[: len(ours)] == ours:
            logger.info('%s: exists with compatible indices — reusing', base_id)
            return base_id
        if stored[: min(len(stored), len(ours))] != ours[: min(len(stored), len(ours))]:
            raise ValueError(
                f'{base_id}: existing baseline indices do not match (seed mismatch?); '
                'rerun with the same --seed as likelihood_misspec.py or pass --overwrite'
            )
        logger.info('%s: exists with %d < %d datasets — extending', base_id, len(stored), len(ours))

    kind = LIK_CONDITIONS[family][0]  # keep the sibling's family-specific baseline kind
    base_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base_dir / 'test.npz', **{k: v[idx] for k, v in source.items()})
    # advi_/laplace_ fits are irrelevant here and dominate the file size — drop them.
    sliced_fit = sliceNpzStreaming(
        src_dir / 'test.fit.npz', idx, drop_prefixes=('advi_', 'laplace_')
    )
    np.savez_compressed(base_dir / 'test.fit.npz', **sliced_fit)
    writeConfig(src_dir, base_dir, base_id, cfg.n_datasets, kind, 0.0, idx, cfg.seed)
    logger.info('%s: baseline written (%d datasets)', base_id, cfg.n_datasets)
    return base_id


def generateCombo(cfg: argparse.Namespace, family: str, size: str) -> list[str]:
    """Write the baseline + condition dirs for one (family, size); returns new data_ids."""
    src_id = f'{size}-{family}-sampled'
    src_dir = DATA_DIR / src_id
    test_path = src_dir / 'test.npz'
    fit_path = src_dir / 'test.fit.npz'
    if not test_path.exists() or not fit_path.exists():
        logger.warning('%s: test.npz or test.fit.npz missing — skipping', src_id)
        return []

    with np.load(test_path, allow_pickle=True) as z:
        source = {k: z[k] for k in z.files}
    B = source['y'].shape[0]
    idx = selectIndices(B, cfg.n_datasets, cfg.seed)
    created = [ensureBaseline(cfg, family, size, src_dir, source, idx)]

    size_pos = DEFAULT_SIZES.index(size)
    for cond_pos, (tag, kind, severity) in enumerate(CONDITIONS):
        data_id = f'{size}-{family}-{tag}'
        out_dir = DATA_DIR / data_id
        created.append(data_id)
        if out_dir.exists() and not cfg.overwrite:
            logger.info('%s: exists — skipping (use --overwrite)', data_id)
            continue
        if out_dir.exists():
            # regenerated X/y invalidates any fits of the previous data — remove them so
            # stale per-index fit files cannot be reintegrated against the new datasets
            stale = [p for p in [out_dir / 'test.fit.npz'] if p.exists()]
            stale += sorted((out_dir / 'fits').glob('*.npz'))
            for p in stale:
                p.unlink()
            if stale:
                logger.warning('%s: removed %d stale fit file(s)', data_id, len(stale))
        batch = {
            k: v[idx].copy()
            for k, v in source.items()
            if k not in STAT_KEYS and not k.startswith(FIT_PREFIXES)
        }
        # 100-offset on the family token keeps streams disjoint from likelihood_misspec.
        rng = np.random.default_rng([cfg.seed, 100 + FAMILY_IDS[family], size_pos, cond_pos])
        regenerateXY(batch, family, kind, severity, rng)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / 'test.npz', **batch)
        writeConfig(
            src_dir,
            out_dir,
            data_id,
            cfg.n_datasets,
            kind,
            severity,
            idx,
            cfg.seed,
            target='predictors',
        )
        logger.info(
            '%s: written (%s severity=%g, %d datasets)', data_id, kind, severity, cfg.n_datasets
        )
    return created


# ---------------------------------------------------------------------------
# Fit campaign
# ---------------------------------------------------------------------------


def printCommands(families: list[str], sizes: list[str], n_datasets: int) -> None:
    """Print the NUTS fit campaign, one size at a time (baseline dirs need no fits)."""
    for size in sizes:
        print(f'\n# --- {size} ---')
        for family in families:
            for tag, _, _ in CONDITIONS:
                print(
                    f'sbatch --array=0-{n_datasets - 1} scripts/fit-nuts.sh --data_id {size}-{family}-{tag}'
                )
        print('# after all fits of this size finished (from metabeta/simulation/):')
        for family in families:
            for tag, _, _ in CONDITIONS:
                print(
                    f'uv run python fit.py --size {size} --family {FAMILY_IDS[family]} '
                    f'--ds_type {tag} --reintegrate'
                )


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)

    if cfg.print_commands:
        printCommands(cfg.families, cfg.sizes, cfg.n_datasets)
        return

    created = []
    for family in cfg.families:
        for size in cfg.sizes:
            created += generateCombo(cfg, family, size)
    if not created:
        logger.error('No data dirs generated.')
        return

    print(f'\nGenerated/verified {len(created)} data dirs:')
    for data_id in created:
        print(f'  {data_id}')
    print('\nNUTS fit campaign (baseline dirs reuse existing fits):')
    printCommands(cfg.families, cfg.sizes, cfg.n_datasets)


if __name__ == '__main__':
    main()
