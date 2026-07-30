"""Likelihood misspecification: regenerate test outcomes under contaminated likelihoods.

Takes existing sampled test datasets, keeps predictors/groups and all generating parameters,
and regenerates y under a misspecified conditional outcome distribution with a severity knob:

    Normal    → Student-t errors        y = η + σ_ε·t_df,        df ∈ {10, 5, 3}
    Poisson   → Negative Binomial       μ = exp(η), Var = μ(1+μ/φ), φ ∈ {10, 3, 1}
    Bernoulli → latent logit noise      y ~ Bern(expit(η + σ_c·ε)), σ_c ∈ {0.5, 1, 2}

(Beta-Binomial with a single trial is exactly Bernoulli — mean-matched overdispersion is
undetectable in binary data — hence the latent-noise contamination, the binary analog of
overdispersion.)  The stored ``likelihood_family`` and prior hyperparameters are left
unchanged: they define the (now misspecified) model that both metabeta and NUTS fit.
Normal outcomes are re-standardized to unit SD exactly as in ``Simulator.sample``, keeping
parameters and prior context coherent, so the stressor is tail shape rather than scale.

Each (size, family) source produces standard-format data dirs under outputs/data/:

    {size}-{fam}-misbase        severity-0 baseline: slice of the original test.npz AND its
                                test.fit.npz (existing NUTS fits reused — zero refits)
    {size}-{fam}-{tag}          one dir per condition (student3, negbin1, latent20, ...),
                                test.npz only — NUTS fits to be produced with fit.py

The three-token data_id keeps the whole toolchain working unmodified (fit-nuts.sh, fit.py,
precompute.py).  Selected source indices are shared across conditions within a (size, family)
and recorded in each config.yaml (``misspec_orig_indices``).

Usage (from repo root):
    uv run python experiments/simulation/likelihood_misspec.py --families n --sizes small --n_datasets 4
    uv run python experiments/simulation/likelihood_misspec.py --n_datasets 32
    uv run python experiments/simulation/likelihood_misspec.py --print_commands

After generation, produce NUTS fits per condition dir (cluster):
    sbatch --array=0-31 scripts/fit-nuts.sh --data_id small-n-student3
then reintegrate (from metabeta/simulation/):
    uv run python fit.py --size small --family 0 --ds_type student3 --reintegrate
and optionally precompute analytical stats (from metabeta/analytical/):
    uv run python precompute.py --size small --family 0 --ds_type student3 --partition test
"""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit

from metabeta.simulation.simulator import SCALE_PARAMS, SCALE_HYPERPARAMS
from metabeta.utils.families import POISSON_ETA_CLIP_MAX
from metabeta.utils.logger import setupLogging
from metabeta.utils.experiments import DATA_DIR

logger = logging.getLogger(__name__)

FAMILY_IDS = {'n': 0, 'b': 1, 'p': 2}
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']
DEFAULT_FAMILIES = ['n', 'b', 'p']
BASE_TAG = 'misbase'

# family letter → (kind, [(ds_type tag, severity)]); severity is df / φ / σ_c.
CONDITIONS: dict[str, tuple[str, list[tuple[str, float]]]] = {
    'n': ('student', [('student10', 10.0), ('student5', 5.0), ('student3', 3.0)]),
    'p': ('negbin', [('negbin10', 10.0), ('negbin3', 3.0), ('negbin1', 1.0)]),
    'b': ('latent', [('latent05', 0.5), ('latent10', 1.0), ('latent20', 2.0)]),
}

# y-dependent analytical stats — stale after regeneration, recompute with precompute.py.
STAT_KEYS = (
    'beta_est',
    'sigma_rfx_est',
    'blup_est',
    'blup_var',
    'sigma_eps_est',
    'phi_pearson',
    'Psi',
)
FIT_PREFIXES = ('nuts_', 'advi_', 'laplace_')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate likelihood-misspecified test sets from sampled test data.')
    parser.add_argument('--families', nargs='+', default=DEFAULT_FAMILIES, choices=DEFAULT_FAMILIES, help='likelihood families (letters)')
    parser.add_argument('--sizes', nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES, help='size regimes')
    parser.add_argument('--n_datasets', type=int, default=32, help='datasets per condition (subset of the 512 test datasets)')
    parser.add_argument('--seed', type=int, default=0, help='seed for index selection and y regeneration')
    parser.add_argument('--overwrite', action='store_true', help='regenerate dirs that already exist')
    parser.add_argument('--print_commands', action='store_true', help='only print the NUTS fit campaign commands and exit')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Streaming npz slicing (baseline dirs)
# ---------------------------------------------------------------------------


def _readHeader(f) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f'unsupported npy version: {version}')


def sliceNpzStreaming(
    path: Path,
    idx: np.ndarray,
    drop_prefixes: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Slice every array in an npz along axis 0 without decompressing full arrays.

    Fit files hold keys with tens of GB decompressed (e.g. huge nuts_rfx); this streams
    each entry row by row and keeps only the selected rows, bounding peak memory by the
    slice size.  ``idx`` holds unique positions in any order; rows are returned in that order.
    """
    want = {int(orig): new for new, orig in enumerate(idx)}
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith('.npy'):
                continue
            key = name[:-4]
            if key.startswith(drop_prefixes):
                continue
            with zf.open(name) as f:
                shape, fortran, dtype = _readHeader(f)
                if fortran:
                    raise ValueError(f'{path}:{key} is Fortran-ordered; streaming assumes C order')
                last = int(idx.max())
                if last >= shape[0]:
                    raise ValueError(f'{path}:{key} has {shape[0]} rows; cannot take idx {last}')
                row_shape = shape[1:]
                row_bytes = int(dtype.itemsize * np.prod(row_shape, dtype=np.int64))
                dest = np.empty((len(idx), *row_shape), dtype=dtype)
                for i in range(last + 1):
                    buf = f.read(row_bytes)
                    if len(buf) != row_bytes:
                        raise ValueError(f'{path}:{key} truncated at row {i}')
                    if i in want:
                        dest[want[i]] = np.frombuffer(buf, dtype=dtype).reshape(row_shape)
                out[key] = dest
    return out


# ---------------------------------------------------------------------------
# Outcome regeneration
# ---------------------------------------------------------------------------


def linearPredictorPadded(batch: dict[str, np.ndarray], i: int) -> np.ndarray:
    """η = Xβ + Z·b for dataset ``i`` of a padded batch, over its n active rows."""
    n = int(batch['n'][i])
    d = int(batch['d'][i])
    q = int(batch['q'][i])
    X = batch['X'][i, :n, :d].astype(np.float64)
    groups = batch['groups'][i, :n].astype(np.int64)
    ffx = batch['ffx'][i, :d].astype(np.float64)
    rfx = batch['rfx'][i, :, :q].astype(np.float64)
    return X @ ffx + (X[:, :q] * rfx[groups]).sum(-1)


def _rescaleDataset(batch: dict[str, np.ndarray], i: int, sd: float) -> None:
    """Divide dataset ``i``'s scale-bearing (hyper)parameters by ``sd`` (Simulator.sample)."""
    for key in SCALE_PARAMS | SCALE_HYPERPARAMS:
        if key in batch:
            batch[key][i] = batch[key][i] / sd
    batch['sd_y'][i] = batch['sd_y'][i] * sd
    if 'r_squared' in batch:
        batch['r_squared'][i] = 1.0 - float(batch['sigma_eps'][i]) ** 2


def regenerateY(
    batch: dict[str, np.ndarray],
    kind: str,
    severity: float,
    rng: np.random.Generator,
) -> None:
    """Regenerate every dataset's outcomes in place under the contaminated likelihood."""
    B = batch['y'].shape[0]
    for i in range(B):
        n = int(batch['n'][i])
        eta = linearPredictorPadded(batch, i)
        if kind == 'student':
            sigma_eps = float(batch['sigma_eps'][i])
            y = eta + sigma_eps * rng.standard_t(severity, size=n)
            sd = max(float(y.std()), 1e-6)
            y /= sd
            _rescaleDataset(batch, i, sd)
        elif kind == 'negbin':
            mu = np.exp(np.minimum(eta, POISSON_ETA_CLIP_MAX))
            p = severity / (severity + mu)
            y = rng.negative_binomial(severity, p).astype(np.float64)
        elif kind == 'latent':
            prob = expit(eta + severity * rng.standard_normal(n))
            y = rng.binomial(1, prob).astype(np.float64)
        else:
            raise ValueError(f'unknown misspecification kind: {kind}')
        batch['y'][i, :n] = y.astype(batch['y'].dtype)
        batch['y'][i, n:] = 0.0


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
) -> None:
    with open(src_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['data_id'] = data_id
    cfg['bs_test'] = n_datasets
    cfg['misspec_kind'] = kind
    cfg['misspec_severity'] = severity
    cfg['misspec_source'] = src_dir.name
    cfg['misspec_orig_indices'] = [int(j) for j in idx]
    cfg['misspec_seed'] = seed
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def selectIndices(B: int, n_datasets: int, seed: int) -> np.ndarray:
    """First ``n_datasets`` of a fixed permutation — deliberately NOT sorted, so smaller
    selections are prefixes of larger ones and their per-index NUTS fits stay valid."""
    if n_datasets > B:
        raise ValueError(f'n_datasets={n_datasets} exceeds test-set size {B}')
    rng = np.random.default_rng(seed)
    return rng.permutation(B)[:n_datasets]


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
    kind, conditions = CONDITIONS[family]
    created = []

    # --- baseline: slice of the original data + its NUTS fits (severity 0, no refits)
    base_id = f'{size}-{family}-{BASE_TAG}'
    base_dir = DATA_DIR / base_id
    if base_dir.exists() and not cfg.overwrite:
        logger.info('%s: exists — skipping (use --overwrite)', base_id)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base_dir / 'test.npz', **{k: v[idx] for k, v in source.items()})
        # advi_/laplace_ fits are irrelevant here and dominate the file size — drop them.
        sliced_fit = sliceNpzStreaming(fit_path, idx, drop_prefixes=('advi_', 'laplace_'))
        np.savez_compressed(base_dir / 'test.fit.npz', **sliced_fit)
        writeConfig(src_dir, base_dir, base_id, cfg.n_datasets, kind, 0.0, idx, cfg.seed)
        logger.info('%s: baseline written (%d datasets)', base_id, cfg.n_datasets)
    created.append(base_id)

    # --- contaminated conditions: regenerated y, fresh NUTS fits required
    size_pos = DEFAULT_SIZES.index(size)
    for cond_pos, (tag, severity) in enumerate(conditions):
        data_id = f'{size}-{family}-{tag}'
        out_dir = DATA_DIR / data_id
        created.append(data_id)
        if out_dir.exists() and not cfg.overwrite:
            logger.info('%s: exists — skipping (use --overwrite)', data_id)
            continue
        if out_dir.exists():
            # regenerated y invalidates any fits of the previous data — remove them so
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
        rng = np.random.default_rng([cfg.seed, FAMILY_IDS[family], size_pos, cond_pos])
        regenerateY(batch, kind, severity, rng)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / 'test.npz', **batch)
        writeConfig(src_dir, out_dir, data_id, cfg.n_datasets, kind, severity, idx, cfg.seed)
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
            _, conditions = CONDITIONS[family]
            for tag, _ in conditions:
                print(
                    f'sbatch --array=0-{n_datasets - 1} scripts/fit-nuts.sh --data_id {size}-{family}-{tag}'
                )
        print('# after all fits of this size finished (from metabeta/simulation/):')
        for family in families:
            _, conditions = CONDITIONS[family]
            for tag, _ in conditions:
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
