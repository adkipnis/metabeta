"""MB↔NUTS agreement and posterior quality under predictor (design) misspecification.

Companion to experiments/simulation/ood_design.py, which replaces test designs with
draws from sources outside the training simulator — heavy-tailed marginals (Student-t,
ν ∈ {2, 1.5, 1}), Clayton tail dependence (Kendall τ ∈ {0.3, 0.6, 0.9}), and longitudinal
growth-curve designs — and regenerates y under the ORIGINAL likelihood.  Unlike likelihood
misspecification, the fitted model remains correctly specified, so NUTS is an exact
reference and the generating parameters are the true ones: any metabeta↔NUTS divergence or
calibration loss isolates the amortization gap on out-of-distribution designs.

Tables mirror experiments/evaluation/likelihood_misspec.py:

  1. MB↔NUTS agreement per condition (r, σ-ratio, rank-MAD, ΔLOO-NLL; median ± MAD over
     the NUTS-converged subset, paired).
  2. Quality vs the generating parameters per condition (EACE / NRMSE / cov90, global and
     local, plus LOO-NLL) for MB, refinements, and NUTS — here directly interpretable
     because the model is correctly specified.

Usage (from repo root):
    uv run python experiments/evaluation/ood_design.py --family n
    uv run python experiments/evaluation/ood_design.py --family n --sizes small --per_size
"""

import argparse
import logging
import sys
from pathlib import Path

from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.sampling import setSeed
from metabeta.utils.experiments import RESULTS_DIR, REPO_ROOT
from metabeta.utils.posterior_eval import posthocDefaults, validMethods

# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS  # noqa: E402

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM
from likelihood_misspec import (
    DEFAULT_SIZES,
    FAMILY_NAMES,
    _poolAgree,
    _poolEntries,
    agreementRows,
    collectCondition,
    qualityRows,
    renderAgreementMd,
    renderAgreementTex,
    renderQualityMd,
    renderQualityTex,
)

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR

# ordered (ds_type tag, display label); baseline first, then the three dials.
CONDITIONS: list[tuple[str, str]] = [
    ('misbase', 'baseline'),
    ('xt2', 'X~t(ν=2)'),
    ('xt15', 'X~t(ν=1.5)'),
    ('xt1', 'X~Cauchy'),
    ('clay3', 'Clayton τ=0.3'),
    ('clay6', 'Clayton τ=0.6'),
    ('clay9', 'Clayton τ=0.9'),
    ('xlong', 'longitudinal'),
]


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='MB↔NUTS agreement and posterior quality under predictor misspecification.',
    )
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='ds_type tags to evaluate (default: all eight)')
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
            raise KeyError(f'unknown condition tags: {unknown}')
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
        logger.error('No conditions evaluated.')
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
    print(
        '\n=== MB↔NUTS agreement by predictor-misspecification condition (median ± MAD, converged) ===\n'
    )
    print(agree_md)

    rows_quality = qualityRows(pooled_entries, pooled_agree, labels, methods)
    quality_md = renderQualityMd(rows_quality, dp=cfg.decimals)
    print('\n=== Quality vs generating parameters by condition (converged subset) ===\n')
    print(quality_md)

    md = [
        f'# Predictor misspecification ({family})\n',
        f'Sizes: {", ".join(sizes)}. Reference: NUTS ({cfg.convergence_mode}). '
        'All metrics on the NUTS-converged subset (paired). The fitted model is correctly '
        'specified under every condition (only the design distribution shifts), so NUTS is '
        'an exact reference and divergence from it isolates the amortization gap.\n',
        '## MB↔NUTS agreement by condition (median ± MAD over converged datasets)\n',
        agree_md,
        '',
        '## Quality vs generating parameters by condition\n',
        'Global (ffx, σ_rfx' + (', σ_ε' if lf == 0 else '') + ') and local (rfx) parameters; '
        'EACE→0 calibrated, cov90 nominal 0.900, LOO-NLL mean per dataset. The generating '
        'parameters are the true parameters of the fitted model, so degradation here is '
        'attributable to the approximation, not the model.\n',
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

    stem = f'ood_design_{family}'
    (outdir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    (outdir / f'{stem}.tex').write_text(
        renderAgreementTex(rows_agree, dp=cfg.decimals)
        + '\n'
        + renderQualityTex(rows_quality, dp=cfg.decimals)
    )
    logger.info('Saved tables to %s', outdir / f'{stem}.md')


if __name__ == '__main__':
    main()
