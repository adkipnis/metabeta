"""Prior misspecification: rewrite the stored prior of existing test sets.

The prior counterpart of experiments/simulation/likelihood_misspec.py.  That script keeps the
prior and contaminates the outcomes; this one keeps the outcomes and contaminates the prior.
Predictors, groups, outcomes and every generating parameter are copied over untouched - only
the stored hyperparameters that define the fitted model are rewritten:

    τ×c         all prior scales (τ_ffx, τ_rfx, τ_ε) multiplied by c ∈ {1/3, 3}
    ν+kτ        ν_ffx shifted by k prior SDs, k ∈ {1, 2} (location, not scale)
    family+1    prior families rotated (Normal<->Student-t, Half-Normal->Half-Student-t->Exp.)

and, outside the trained range of the hyperparameters (opt-in via --conditions):

    oosA        every scale set to twice the largest stored value of the training sets
    oosB        ν_ffx shifted by 5 prior SDs
    oosC        every scale set to half the smallest stored value of the training sets

Because metabeta reads those fields as posterior context and ``utils/pymc.buildPymc`` reads
them to construct the PyMC model, writing them into a data dir is enough to make *both*
inference paths fit the same misspecified prior - no changes to fit.py or fit-ref.sh.  That
is the whole point of doing this at generation time rather than perturbing in memory: without
a matching NUTS refit there is no gold standard to compare the degradation against.

The generating parameters still come from the *correct* prior, so a perturbed condition is a
genuine modelling error, and the question is whether metabeta degrades in step with NUTS.

Each (size, family) source produces standard-format data dirs under outputs/data/:

    {size}-{fam}-priorbase      unperturbed baseline: slice of the original test.npz AND its
                                test.nuts.npz (existing NUTS fits reused - zero refits)
    {size}-{fam}-{tag}          one dir per condition (tau033, tau3, mu1, mu2, famrot),
                                test.npz only - NUTS fits to be produced with fit.py

Analytical stats are a function of the prior, so they are dropped from every dir (baseline
included, to keep the conditions comparable): either rerun precompute.py on all of them, or
let the model recompute them live.  Selected source indices are shared across conditions
within a (size, family) and recorded in each config.yaml (``prior_orig_indices``).

Usage (from repo root):
    uv run python experiments/simulation/prior_misspec.py --families n --sizes small --n_datasets 4
    uv run python experiments/simulation/prior_misspec.py --n_datasets 32
    uv run python experiments/simulation/prior_misspec.py --print_commands
    uv run python experiments/simulation/prior_misspec.py --sizes medium --n_datasets 128 \
        --conditions oosA oosB --base_tag oosbase

After generation, produce NUTS fits per condition dir (cluster):
    scripts/fit-ref.sh --method nuts --level 2 --n_datasets 32 --data_id small-n-tau3
then check and reintegrate:
    uv run python metabeta/simulation/check.py --partition test --data_id small-n-tau3
and optionally precompute analytical stats (from metabeta/analytical/):
    uv run python precompute.py --size small --family 0 --ds_type tau3 --partition test
"""

import argparse
import logging
from typing import Literal, NamedTuple

import numpy as np
import yaml

from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.logger import setupLogging
from metabeta.utils.experiments import DATA_DIR
from metabeta.utils.fits import REFERENCE_TAG, availableFits, fitPath

# sibling experiment script (this directory is sys.path[0] at run time); the npz slicing,
# index selection and stale-key lists are identical for both misspecification studies.
from likelihood_misspec import (
    DEFAULT_FAMILIES,
    DEFAULT_SIZES,
    FIT_PREFIXES,
    STAT_KEYS,
    copyReferenceFit,
    selectIndices,
)

logger = logging.getLogger(__name__)

BASE_TAG = 'priorbase'


class Condition(NamedTuple):
    """How one ds_type rewrites the stored prior.

    Scales are multiplied by ``scale``, unless ``pin`` names an end of the trained range: then
    every scale is set to ``scale`` times that end (see trainedRange), the same for all datasets.
    """

    scale: float = 1.0
    shift: float = 0.0  # ν_ffx shift in units of τ_ffx
    rotate: bool = False  # rotate the prior families
    pin: Literal['min', 'max'] | None = None


# ds_type tag -> condition; kept in step with the display labels in
# experiments/evaluation/prior_misspec.py.
CONDITIONS: dict[str, Condition] = {
    'tau033': Condition(scale=1.0 / 3.0),
    'tau3': Condition(scale=3.0),
    'mu1': Condition(shift=1.0),
    'mu2': Condition(shift=2.0),
    'famrot': Condition(rotate=True),
    'oosA': Condition(scale=2.0, pin='max'),
    'oosB': Condition(shift=5.0),
    'oosC': Condition(scale=0.5, pin='min'),
}
DEFAULT_CONDITIONS = ['tau033', 'tau3', 'mu1', 'mu2', 'famrot']

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
    parser.add_argument('--conditions', nargs='+', default=DEFAULT_CONDITIONS, choices=list(CONDITIONS), help='ds_type tags to generate')
    parser.add_argument('--base_tag', type=str, default=BASE_TAG, help='ds_type tag of the unperturbed baseline')
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


def trainedRange(family: str) -> dict[str, tuple[float, float]]:
    """Smallest and largest stored value of each scale hyperparameter over the family's sampled
    test and validation sets of every size, a proxy for the scales the network was trained on.

    Stored scales are standardized (divided by sd(y) for Gaussian outcomes, shrunk by the
    linear-predictor cap otherwise), so this range differs from the drawn ranges in
    simulation/prior.py: for Gaussian outcomes it is much wider.
    """
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for size in DEFAULT_SIZES:
        for partition in ('test', 'valid'):
            path = DATA_DIR / f'{size}-{family}-sampled' / f'{partition}.npz'
            if not path.exists():
                continue
            with np.load(path) as z:
                for key in SCALE_KEYS:
                    if key in z.files:
                        v = z[key][z[key] > 0]  # padded entries are 0
                        lo[key] = min(lo.get(key, np.inf), float(v.min()))
                        hi[key] = max(hi.get(key, -np.inf), float(v.max()))
    if not hi:
        raise FileNotFoundError(f'no sampled test or validation set for family {family}')
    return {key: (lo[key], hi[key]) for key in hi}


def pinnedScales(cond: Condition, trained: dict[str, tuple[float, float]]) -> dict[str, float]:
    """The value every scale hyperparameter is set to under a pinned condition."""
    end = 1 if cond.pin == 'max' else 0
    return {key: cond.scale * bounds[end] for key, bounds in trained.items()}


def perturbPrior(
    batch: dict[str, np.ndarray],
    cond: Condition,
    pinned: dict[str, float] | None = None,
) -> None:
    """Rewrite every dataset's stored prior hyperparameters in place.

    Padded columns stay zero under every operation (0·c = 0, 0 + k·0 = 0, and pinning skips
    them), so the active/inactive structure of the batch is preserved.  The shift is applied
    before the rescale so that ``k`` always counts *original* prior SDs.
    """
    if cond.shift != 0.0:
        batch['nu_ffx'] = batch['nu_ffx'] + cond.shift * batch['tau_ffx']
    for key in SCALE_KEYS:
        if key not in batch:
            continue
        if pinned is not None:
            batch[key] = np.where(batch[key] > 0, pinned[key], 0.0).astype(batch[key].dtype)
        elif cond.scale != 1.0:
            batch[key] = batch[key] * cond.scale
    if cond.rotate:
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
    cond: Condition,
    pinned: dict[str, float] | None,
    idx: np.ndarray,
    seed: int,
) -> None:
    with open(src_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['data_id'] = data_id
    cfg['bs_test'] = n_datasets
    cfg['prior_misspec_tag'] = tag
    cfg['prior_misspec_scale'] = cond.scale
    cfg['prior_misspec_shift'] = cond.shift
    cfg['prior_misspec_rotate_family'] = cond.rotate
    if pinned is not None:
        cfg['prior_misspec_pin'] = cond.pin
        cfg['prior_misspec_pinned'] = pinned
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


def generateCombo(
    cfg: argparse.Namespace,
    family: str,
    size: str,
    trained: dict[str, tuple[float, float]] | None,
) -> list[str]:
    """Write the baseline + condition dirs for one (family, size); returns new data_ids."""
    src_id = f'{size}-{family}-sampled'
    src_dir = DATA_DIR / src_id
    test_path = src_dir / 'test.npz'
    if not test_path.exists() or not fitPath(test_path, REFERENCE_TAG).exists():
        logger.warning('%s: test.npz or its %s fit missing - skipping', src_id, REFERENCE_TAG)
        return []

    with np.load(test_path, allow_pickle=True) as z:
        source = {k: z[k] for k in z.files}
    B = source['y'].shape[0]
    idx = selectIndices(B, cfg.n_datasets, cfg.seed)
    created = []

    # --- baseline: correct prior, slice of the original data + its NUTS fits (no refits)
    base_id = f'{size}-{family}-{cfg.base_tag}'
    base_dir = DATA_DIR / base_id
    if base_dir.exists() and not cfg.overwrite:
        logger.info('%s: exists - skipping (use --overwrite)', base_id)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base_dir / 'test.npz', **_dropStaleKeys(source, idx))
        copyReferenceFit(test_path, base_dir / 'test.npz', idx)
        writeConfig(
            src_dir,
            base_dir,
            base_id,
            cfg.n_datasets,
            cfg.base_tag,
            Condition(),
            None,
            idx,
            cfg.seed,
        )
        logger.info('%s: baseline written (%d datasets)', base_id, cfg.n_datasets)
    created.append(base_id)

    # --- perturbed conditions: rewritten prior, fresh NUTS fits required
    for tag in cfg.conditions:
        cond = CONDITIONS[tag]
        pinned = pinnedScales(cond, trained) if cond.pin is not None else None
        data_id = f'{size}-{family}-{tag}'
        out_dir = DATA_DIR / data_id
        created.append(data_id)
        if out_dir.exists() and not cfg.overwrite:
            logger.info('%s: exists - skipping (use --overwrite)', data_id)
            continue
        if out_dir.exists():
            # a rewritten prior invalidates any fits of the previous data - remove them so
            # stale per-index fit files cannot be reintegrated against the new prior
            stale = [fitPath(out_dir / 'test.npz', t) for t in availableFits(out_dir / 'test.npz')]
            stale += sorted((out_dir / 'fits').glob('*.npz'))
            for p in stale:
                p.unlink()
            if stale:
                logger.warning('%s: removed %d stale fit file(s)', data_id, len(stale))
        batch = _dropStaleKeys(source, idx)
        perturbPrior(batch, cond, pinned)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / 'test.npz', **batch)
        writeConfig(src_dir, out_dir, data_id, cfg.n_datasets, tag, cond, pinned, idx, cfg.seed)
        logger.info(
            '%s: written (%s, pinned=%s, %d datasets)', data_id, cond, pinned, cfg.n_datasets
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
    """Print the NUTS ladder campaign, one size at a time (baseline dirs need no fits)."""
    for size in sizes:
        print(f'\n# --- {size} ---')
        for family in families:
            for tag in conditions:
                print(
                    f'scripts/fit-ref.sh --method nuts --level 2 '
                    f'--n_datasets {n_datasets} --data_id {size}-{family}-{tag}'
                )
        print('# after all fits of this size finished (checks the fits, then reintegrates):')
        for family in families:
            for tag in conditions:
                print(
                    f'uv run python metabeta/simulation/check.py --partition test '
                    f'--data_id {size}-{family}-{tag}'
                )


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    if cfg.print_commands:
        printCommands(cfg.families, cfg.sizes, cfg.conditions, cfg.n_datasets)
        return

    created = []
    for family in cfg.families:
        pins = any(CONDITIONS[tag].pin is not None for tag in cfg.conditions)
        trained = trainedRange(family) if pins else None
        if trained is not None:
            logger.info('%s: trained scale range %s', family, trained)
        for size in cfg.sizes:
            created += generateCombo(cfg, family, size, trained)
    if not created:
        logger.error('No data dirs generated.')
        return

    print(f'\nGenerated/verified {len(created)} data dirs:')
    for data_id in created:
        print(f'  {data_id}')
    print('\nNUTS fit campaign (baseline dirs reuse existing fits):')
    printCommands(cfg.families, cfg.sizes, cfg.conditions, cfg.n_datasets)


if __name__ == '__main__':
    main()
