"""Prior misspecification: rewrite the stored prior of existing test sets.

The prior counterpart of experiments/simulation/likelihood_misspec.py.  That script keeps the
prior and contaminates the outcomes; this one keeps the outcomes and contaminates the prior.
Predictors, groups, outcomes and every generating parameter are copied over untouched — only
the stored hyperparameters that define the fitted model are rewritten:

    τ×c         all prior scales (τ_ffx, τ_rfx, τ_ε) multiplied by c ∈ {1/3, 3}
    ν+kτ        ν_ffx shifted by k prior SDs, k ∈ {1, 2} (location, not scale)
    family+1    prior families rotated (Normal↔Student-t, Half-Normal→Half-Student-t→Exp.)

Because metabeta reads those fields as posterior context and ``utils/pymc.buildPymc`` reads
them to construct the PyMC model, writing them into a data dir is enough to make *both*
inference paths fit the same misspecified prior — no changes to fit.py or fit-nuts.sh.  That
is the whole point of doing this at generation time rather than perturbing in memory: without
a matching NUTS refit there is no gold standard to compare the degradation against.

The generating parameters still come from the *correct* prior, so a perturbed condition is a
genuine modelling error, and the question is whether metabeta degrades in step with NUTS.

Each (size, family) source produces standard-format data dirs under outputs/data/:

    {size}-{fam}-priorbase      unperturbed baseline: slice of the original test.npz AND its
                                test.fit.npz (existing NUTS fits reused — zero refits)
    {size}-{fam}-{tag}          one dir per condition (tau033, tau3, mu1, mu2, famrot),
                                test.npz only — NUTS fits to be produced with fit.py

Analytical stats are a function of the prior, so they are dropped from every dir (baseline
included, to keep the conditions comparable): either rerun precompute.py on all of them, or
let the model recompute them live.  Selected source indices are shared across conditions
within a (size, family) and recorded in each config.yaml (``prior_orig_indices``).

Usage (from repo root):
    uv run python experiments/simulation/prior_misspec.py --families n --sizes small --n_datasets 4
    uv run python experiments/simulation/prior_misspec.py --n_datasets 32
    uv run python experiments/simulation/prior_misspec.py --print_commands

After generation, produce NUTS fits per condition dir (cluster):
    sbatch --array=0-31 scripts/fit-nuts.sh --data_id small-n-tau3
then reintegrate (from metabeta/simulation/):
    uv run python fit.py --size small --family 0 --ds_type tau3 --reintegrate
and optionally precompute analytical stats (from metabeta/analytical/):
    uv run python precompute.py --size small --family 0 --ds_type tau3 --partition test
"""

import argparse
import logging

import numpy as np
import yaml

from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.logger import setupLogging
from metabeta.utils.experiments import DATA_DIR

# sibling experiment script (this directory is sys.path[0] at run time); the npz slicing,
# index selection and stale-key lists are identical for both misspecification studies.
from likelihood_misspec import (
    DEFAULT_FAMILIES,
    DEFAULT_SIZES,
    FAMILY_IDS,
    FIT_PREFIXES,
    STAT_KEYS,
    selectIndices,
    sliceNpzStreaming,
)

logger = logging.getLogger(__name__)

BASE_TAG = 'priorbase'

# ds_type tag → (τ multiplier, ν_ffx shift in units of τ_ffx, rotate prior families).
# Kept in step with the display labels in experiments/evaluation/prior_misspec.py.
CONDITIONS: dict[str, tuple[float, float, bool]] = {
    'tau033': (1.0 / 3.0, 0.0, False),
    'tau3': (3.0, 0.0, False),
    'mu1': (1.0, 1.0, False),
    'mu2': (1.0, 2.0, False),
    'famrot': (1.0, 0.0, True),
}

# hyperparameters that define the fitted prior; everything else is copied verbatim
SCALE_KEYS = ('tau_ffx', 'tau_rfx', 'tau_eps')
FAMILY_KEYS = (
    ('family_ffx', len(FFX_FAMILIES)),
    ('family_sigma_rfx', len(SIGMA_FAMILIES)),
    ('family_sigma_eps', len(SIGMA_FAMILIES)),
)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate prior-misspecified test sets from sampled test data.')
    parser.add_argument('--families', nargs='+', default=DEFAULT_FAMILIES, choices=DEFAULT_FAMILIES, help='likelihood families (letters)')
    parser.add_argument('--sizes', nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES, help='size regimes')
    parser.add_argument('--conditions', nargs='+', default=None, choices=list(CONDITIONS), help='ds_type tags to generate (default: all)')
    parser.add_argument('--n_datasets', type=int, default=32, help='datasets per condition (subset of the 512 test datasets)')
    parser.add_argument('--seed', type=int, default=0, help='seed for index selection')
    parser.add_argument('--overwrite', action='store_true', help='regenerate dirs that already exist')
    parser.add_argument('--print_commands', action='store_true', help='only print the NUTS fit campaign commands and exit')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Prior rewriting
# ---------------------------------------------------------------------------


def perturbPrior(
    batch: dict[str, np.ndarray],
    scale: float,
    shift: float,
    rotate: bool,
) -> None:
    """Rewrite every dataset's stored prior hyperparameters in place.

    Padded columns stay zero under both operations (0·c = 0 and 0 + k·0 = 0), so the
    active/inactive structure of the batch is preserved.  The shift is applied before the
    rescale so that ``k`` always counts *original* prior SDs.
    """
    if shift != 0.0:
        batch['nu_ffx'] = batch['nu_ffx'] + shift * batch['tau_ffx']
    if scale != 1.0:
        for key in SCALE_KEYS:
            if key in batch:
                batch[key] = batch[key] * scale
    if rotate:
        for key, n_families in FAMILY_KEYS:
            if key in batch:
                batch[key] = (batch[key] + 1) % n_families


# ---------------------------------------------------------------------------
# Directory writing
# ---------------------------------------------------------------------------


def writeConfig(
    src_dir,
    out_dir,
    data_id: str,
    n_datasets: int,
    tag: str,
    scale: float,
    shift: float,
    rotate: bool,
    idx: np.ndarray,
    seed: int,
) -> None:
    with open(src_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['data_id'] = data_id
    cfg['bs_test'] = n_datasets
    cfg['prior_misspec_tag'] = tag
    cfg['prior_misspec_scale'] = scale
    cfg['prior_misspec_shift'] = shift
    cfg['prior_misspec_rotate_family'] = rotate
    cfg['prior_misspec_source'] = src_dir.name
    cfg['prior_orig_indices'] = [int(j) for j in idx]
    cfg['prior_misspec_seed'] = seed
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _dropStaleKeys(source: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Sliced copy without the prior-dependent analytical stats and any fit arrays."""
    return {
        k: v[idx].copy()
        for k, v in source.items()
        if k not in STAT_KEYS and not k.startswith(FIT_PREFIXES)
    }


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
    conditions = cfg.conditions if cfg.conditions is not None else list(CONDITIONS)
    created = []

    # --- baseline: correct prior, slice of the original data + its NUTS fits (no refits)
    base_id = f'{size}-{family}-{BASE_TAG}'
    base_dir = DATA_DIR / base_id
    if base_dir.exists() and not cfg.overwrite:
        logger.info('%s: exists — skipping (use --overwrite)', base_id)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base_dir / 'test.npz', **_dropStaleKeys(source, idx))
        # advi_/laplace_ fits are irrelevant here and dominate the file size — drop them.
        sliced_fit = sliceNpzStreaming(fit_path, idx, drop_prefixes=('advi_', 'laplace_'))
        np.savez_compressed(base_dir / 'test.fit.npz', **sliced_fit)
        writeConfig(
            src_dir, base_dir, base_id, cfg.n_datasets, BASE_TAG, 1.0, 0.0, False, idx, cfg.seed
        )
        logger.info('%s: baseline written (%d datasets)', base_id, cfg.n_datasets)
    created.append(base_id)

    # --- perturbed conditions: rewritten prior, fresh NUTS fits required
    for tag in conditions:
        scale, shift, rotate = CONDITIONS[tag]
        data_id = f'{size}-{family}-{tag}'
        out_dir = DATA_DIR / data_id
        created.append(data_id)
        if out_dir.exists() and not cfg.overwrite:
            logger.info('%s: exists — skipping (use --overwrite)', data_id)
            continue
        if out_dir.exists():
            # a rewritten prior invalidates any fits of the previous data — remove them so
            # stale per-index fit files cannot be reintegrated against the new prior
            stale = [p for p in [out_dir / 'test.fit.npz'] if p.exists()]
            stale += sorted((out_dir / 'fits').glob('*.npz'))
            for p in stale:
                p.unlink()
            if stale:
                logger.warning('%s: removed %d stale fit file(s)', data_id, len(stale))
        batch = _dropStaleKeys(source, idx)
        perturbPrior(batch, scale, shift, rotate)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / 'test.npz', **batch)
        writeConfig(
            src_dir, out_dir, data_id, cfg.n_datasets, tag, scale, shift, rotate, idx, cfg.seed
        )
        logger.info(
            '%s: written (τ×%g, ν+%gτ, rotate=%s, %d datasets)',
            data_id,
            scale,
            shift,
            rotate,
            cfg.n_datasets,
        )
    return created


# ---------------------------------------------------------------------------
# Fit campaign
# ---------------------------------------------------------------------------


def printCommands(
    families: list[str],
    sizes: list[str],
    conditions: list[str],
    n_datasets: int,
) -> None:
    """Print the NUTS fit campaign, one size at a time (baseline dirs need no fits)."""
    for size in sizes:
        print(f'\n# --- {size} ---')
        for family in families:
            for tag in conditions:
                print(
                    f'sbatch --array=0-{n_datasets - 1} scripts/fit-nuts.sh '
                    f'--data_id {size}-{family}-{tag}'
                )
        print('# after all fits of this size finished (from metabeta/simulation/):')
        for family in families:
            for tag in conditions:
                print(
                    f'uv run python fit.py --size {size} --family {FAMILY_IDS[family]} '
                    f'--ds_type {tag} --reintegrate'
                )


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    conditions = cfg.conditions if cfg.conditions is not None else list(CONDITIONS)

    if cfg.print_commands:
        printCommands(cfg.families, cfg.sizes, conditions, cfg.n_datasets)
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
    printCommands(cfg.families, cfg.sizes, conditions, cfg.n_datasets)


if __name__ == '__main__':
    main()
