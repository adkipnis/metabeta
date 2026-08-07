"""MB↔NUTS agreement and posterior quality under prior misspecification.

Companion to experiments/simulation/prior_misspec.py, which rewrites the stored prior of the
sampled test sets (τ×c, ν+kτ, family rotation) while leaving the data and the generating
parameters untouched.  Both metabeta and NUTS read that prior from the data dir, so both fit
the same misspecified model — the same arrangement as the likelihood-misspecification study,
and the reason the perturbation happens at generation time rather than in memory.

Two dose–response tables, sharing their whole collection and table layer with
experiments/evaluation/likelihood_misspec.py:

  1. MB↔NUTS agreement per condition (r, σ-ratio, rank-MAD, ΔLOO-NLL; median ± MAD over the
     NUTS-converged subset).  Both sides now fit the *same* wrong prior, so stable agreement
     across conditions means MB's extra error is attributable to the misspecification rather
     than to the amortization.
  2. Quality vs the generating parameters per condition (NRMSE / EACE for global and
     local parameters, plus LOO-NLL) for MB, MB+IMH and NUTS.  The generating parameters came
     from the correct prior, so absolute degradation is expected by design — the question is
     whether MB degrades in step with NUTS.

Sizes are pooled per condition; ``--per_size`` adds per-size agreement breakdowns.  Each size
uses its regime-matched checkpoint (BEST_SEEDS from scripts/build_ckpt.py).

Requires the condition dirs and their NUTS fits to exist; generate them first with
experiments/simulation/prior_misspec.py (which prints the fit campaign).

Usage (from repo root):
    uv run python experiments/evaluation/prior_misspec.py --family n
    uv run python experiments/evaluation/prior_misspec.py --family n --sizes small --per_size
    uv run python experiments/evaluation/prior_misspec.py --family b --conditions priorbase tau3
"""

import argparse
import logging
import sys
from pathlib import Path

from metabeta.utils.device import setDevice
from metabeta.utils.experiments import RESULTS_DIR, REPO_ROOT
from metabeta.utils.logger import setupLogging
from metabeta.utils.sampling import setSeed
from metabeta.utils.posterior_eval import posthocDefaults, validMethods

# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS  # noqa: E402

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM

# Collection and the table layer are shared verbatim with the likelihood-misspecification
# study: once the perturbed prior lives in the data dir, a prior condition and a likelihood
# condition are the same object — a data dir with its own NUTS fits.
from likelihood_misspec import (
    FAMILY_NAMES,
    agreementRows,
    alignSizes,
    collectCondition,
    qualityRows,
    renderAgreementMd,
    renderAgreementTex,
    renderQualityMd,
    renderQualityTex,
    _poolAgree,
    _poolEntries,
)

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']

# ds_type tag → display label; severity 0 first.  Tags must match the dirs written by
# experiments/simulation/prior_misspec.py.
CONDITIONS: list[tuple[str, str]] = [
    ('priorbase', 'correct'),
    ('tau033', 'τ×1/3'),
    ('tau3', 'τ×3'),
    ('mu1', 'ν+1τ'),
    ('mu2', 'ν+2τ'),
    ('famrot', 'family+1'),
]


# ---------------------------------------------------------------------------
# CLI / main


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='MB↔NUTS agreement and posterior quality under prior misspecification.',
    )
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='ds_type tags to evaluate (default: all six)')
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--convergence_mode', type=str, default='strict', choices=['liberal', 'strict'])
    parser.add_argument('--methods', type=str, nargs='*', default=None,
                        help='refinements added as extra rows (default: family preset); MB always included')
    parser.add_argument('--per_size', action='store_true',
                        help='additionally print per-size agreement tables (no pooling)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR))
    parser.add_argument('--decimals', type=int, default=3)
    parser.add_argument('--verbosity', type=int, default=1)
    # fmt: on
    return parser.parse_args()


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    family = cfg.family
    lf = LF_FROM_FAM[family]
    posthoc = cfg.methods if cfg.methods is not None else posthocDefaults(lf)
    methods = ['mb'] + validMethods(posthoc, lf)

    conditions = CONDITIONS
    if cfg.conditions is not None:
        known = dict(conditions)
        unknown = [t for t in cfg.conditions if t not in known]
        if unknown:
            raise KeyError(f'unknown condition tags: {unknown} (known: {list(known)})')
        conditions = [(t, known[t]) for t in cfg.conditions]
    labels = dict(conditions)
    sizes = [s for s in cfg.sizes if BEST_SEEDS.get((FAMILY_NAMES[family], s)) is not None]

    per_cond: dict[str, list[dict]] = {}
    for tag, _ in conditions:
        collected = [
            r
            for r in (collectCondition(cfg, family, s, tag, device, methods) for s in sizes)
            if r is not None
        ]
        if collected:
            per_cond[tag] = collected
    if not per_cond:
        logger.error(
            'No conditions evaluated. Generate the data dirs first with '
            'experiments/simulation/prior_misspec.py.'
        )
        return
    # the baseline reuses existing fits and so covers every size immediately, while a perturbed
    # condition only appears once its own NUTS campaign lands — pool like against like
    per_cond, sizes = alignSizes(per_cond)
    if not per_cond:
        logger.error('No size was collected under every condition — nothing comparable to pool.')
        return

    pooled_agree = {tag: _poolAgree(per) for tag, per in per_cond.items()}
    pooled_entries = {
        tag: {
            label: {kind: _poolEntries(per, label, kind) for kind in ('local', 'global')}
            for label in per[0]['entries']
        }
        for tag, per in per_cond.items()
    }

    rows_agree = agreementRows(pooled_agree, labels, methods)
    agree_md = renderAgreementMd(rows_agree, dp=cfg.decimals)
    print('\n=== MB↔NUTS agreement by prior perturbation (median ± MAD, converged) ===\n')
    print(agree_md)

    rows_quality = qualityRows(pooled_entries, pooled_agree, labels, methods)
    quality_md = renderQualityMd(rows_quality, dp=cfg.decimals)
    print('\n=== Quality vs generating parameters by perturbation (converged subset) ===\n')
    print(quality_md)

    md = [
        f'# Prior misspecification ({family})\n',
        f'Sizes: {", ".join(sizes)}. Reference: NUTS ({cfg.convergence_mode}), refit under '
        'each perturbed prior. All metrics on the NUTS-converged subset (paired).\n',
        '## MB↔NUTS agreement by perturbation (median ± MAD over converged datasets)\n',
        agree_md,
        '',
        '## Quality vs generating parameters by perturbation\n',
        'Global (ffx, σ_rfx' + (', σ_ε' if lf == 0 else '') + ') and local (rfx) parameters; '
        'NRMSE→0 recovery, EACE→0 calibrated, LOO-NLL median per dataset. The generating '
        'parameters come from the correct prior, so a perturbed prior is a genuine modelling '
        'error and absolute degradation is expected — the comparison is MB vs NUTS.\n',
        quality_md,
        '',
    ]

    if cfg.per_size:
        for size in sizes:
            sized = {
                tag: _poolAgree([r for r in per if r['agree']['size'][0] == size])
                for tag, per in per_cond.items()
                if any(r['agree']['size'][0] == size for r in per)
            }
            if not sized:
                continue
            tbl = renderAgreementMd(agreementRows(sized, labels, methods), dp=cfg.decimals)
            print(f'\n=== Agreement, {size} only ===\n')
            print(tbl)
            md += [f'## Agreement, {size} only\n', tbl, '']

    stem = f'prior_misspec_{family}'
    (outdir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    (outdir / f'{stem}.tex').write_text(
        renderAgreementTex(rows_agree, dp=cfg.decimals)
        + '\n'
        + renderQualityTex(rows_quality, dp=cfg.decimals)
    )
    logger.info('Saved tables to %s', outdir / f'{stem}.md')


if __name__ == '__main__':
    main()
