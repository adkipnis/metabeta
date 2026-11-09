"""MB<->NUTS agreement along the stress-search axes: one leverage point and the linear-predictor scale.

Companion to experiments/simulation/stress_search.py, which pushes one design value (x_ij = +-k)
or the scale of the linear predictor (Bernoulli sd(η), Gaussian R²) past the training range on
64 sampled medium datasets per family, with NUTS refit per level under the stored prior. The
fitted model stays correctly specified, so NUTS is exact and a gap isolates the amortization
error.

One report per family and axis, baseline first, sharing the collection and table layer of
likelihood_misspec.py and prior_misspec.py:

  1. MB<->NUTS agreement per level (r, σ-ratio, rank-MAD, ΔLOO-NLL; median ± MAD over the
     NUTS-converged subset) with % NUTS-converged.
  2. IMH acceptance and the share of datasets the Api's IMH warnings flag (all datasets).
  3. The stressed parameters per dataset, since the pooled metrics of 1 are dominated by the
     group effects: the coefficient of the stressed predictor on the leverage axis, every
     active global parameter on the scale axis. A dataset is off NUTS when a stressed mean
     lies more than SHIFT_MAX NUTS posterior SDs away or a stressed SD is off by more than a
     factor SD_RATIO_MAX; the off datasets are counted with and without a flag.

The flags are the Api's warnings (metabeta.models.api.imhFlagged), read from each method's own
IMH run: acceptance below IMH_ACCEPT_WARN or min ESS below IMH_ESS_WARN. The random-walk moves
run exactly where that ESS falls below IMH_ESS_WARN, so with the moves a dataset is flagged where
they ran or where its IMH acceptance is low. The per-dataset arrays go to
stress_{family}_{axis}.npz, keyed '{tag}/{key}', with 'idx' the position in the set.

Usage (from repo root):
    uv run python experiments/evaluation/stress_search.py --family n \\
        --methods imhMarginal imhMarginal+moves --n_samples 4000
    uv run python experiments/evaluation/stress_search.py --family p --axes leverage \\
        --methods imhPM imhPM+moves --n_samples 4000
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from tabulate import tabulate

from metabeta.models.api import IMH_ACCEPT_WARN, IMH_ESS_WARN, imhFlagged
from metabeta.posthoc.metropolis import bulkEss
from metabeta.utils.device import setDevice
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR
from metabeta.utils.logger import setupLogging
from metabeta.utils.posterior_eval import (
    IMH_METHODS,
    IMH_MOVES,
    IMH_N_CHAINS,
    posthocDefaults,
    validMethods,
)
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM, METHOD_LABELS
from likelihood_misspec import (
    FAMILY_NAMES,
    agreementRows,
    collectCondition,
    renderAgreementMd,
    renderAgreementTex,
)
from prior_misspec import renderDiagnosticsMd
from real_posterior import _masks, _weightedMean, _weightedStd

logger = logging.getLogger(__name__)

SIZE = 'medium'
BASELINE = ('stressbase', 'baseline')
# ds_type tag -> display label, in the order of experiments/simulation/stress_search.py
AXES: dict[str, dict[str, list[tuple[str, str]]]] = {
    'leverage': {f: [(f'lev{k}', f'|x|={k}') for k in (2, 3, 5, 10, 20, 50)] for f in 'nbp'},
    'scale': {
        'n': [
            ('r295', 'R²=0.95'),
            ('r297', 'R²=0.97'),
            ('r299', 'R²=0.99'),
            ('r2995', 'R²=0.995'),
            ('r2999', 'R²=0.999'),
        ],
        'b': [(f'eta{s}', f'sd(η)={s}') for s in (4, 6, 8, 12, 16)],
        # past sd(η) of about 2 the Poisson simulator clips η at 10, which PyMC does not
        'p': [],
    },
}
SHIFT_MAX = 1.0  # |Δmean| in NUTS posterior SDs
SD_RATIO_MAX = 1.5
# a run's share of changed steps differs from its stored IMH acceptance by at most 0.001 where
# the moves did not run (the first kept step's decision is not in the draws)
ACCEPT_GAP_MOVED = 0.002


def movesRan(proposal: Proposal) -> np.ndarray:
    """(B,) the datasets on which a '+moves' run's random-walk moves ran.

    The run keeps only its IMH acceptance. Where the moves did not run, its draws are the IMH
    chains, whose share of changed steps matches that acceptance and whose min bulk ESS passed
    the moves' gate (IMH_ESS_WARN); where they ran, the draws hold the moves' chains instead.
    """
    samples_g = proposal.samples_g  # (B, S, D)
    B, S, D = samples_g.shape
    x = samples_g.reshape(B, IMH_N_CHAINS, S // IMH_N_CHAINS, D)
    prev, cur = x[:, :, :-1], x[:, :, 1:]
    changed = ((cur != prev) & ~(cur.isnan() & prev.isnan())).any(-1)  # (B, C, S/C - 1)
    accept = proposal.is_results['imh_accept_rate'].double().numpy()
    gap = np.abs(changed.double().mean((1, 2)).numpy() - accept)
    ess = np.nanmin(bulkEss(samples_g, IMH_N_CHAINS).numpy(), axis=1)
    return (gap >= ACCEPT_GAP_MOVED) | (ess < IMH_ESS_WARN)


def stressedShift(stress_col: np.ndarray | None) -> Callable:
    """collectCondition hook: how far each method's stressed parameters sit from NUTS.

    Per dataset, over the stressed parameters (the fixed effect of column ``stress_col``, or
    every active global parameter when it is None): the largest |Δmean| in NUTS posterior
    SDs, and the SD ratio (method / NUTS) furthest from 1. For a '+moves' run, also whether
    its moves ran (``{m}:moved``, see movesRan), which apiFlags reads.
    """

    def hook(
        proposals: dict[str, Proposal], proposal_nuts: Proposal, batch: dict
    ) -> dict[str, np.ndarray]:
        mean_n, std_n = _weightedMean(proposal_nuts), _weightedStd(proposal_nuts)
        keys = ['ffx', 'sigma_rfx'] + (['sigma_eps'] if 'sigma_eps' in mean_n else [])
        B = mean_n['ffx'].shape[0]

        def flat(stats: dict) -> np.ndarray:
            return np.concatenate([stats[k].reshape(B, -1).double().numpy() for k in keys], 1)

        mu_n, sd_n = flat(mean_n), flat(std_n)
        if stress_col is None:
            mask_d, mask_q, _ = _masks(batch)
            sel = np.concatenate(
                [mask_d.numpy(), mask_q.numpy()] + [np.ones((B, 1))] * (len(keys) - 2), 1
            ).astype(bool)
        else:
            sel = np.zeros(mu_n.shape, dtype=bool)
            sel[np.arange(B), stress_col] = True
        d_pad, q_pad = mean_n['ffx'].shape[1], mean_n['sigma_rfx'].shape[1]
        out = {'idx': np.arange(B), 'd_pad': np.full(B, d_pad), 'q_pad': np.full(B, q_pad)}
        for m, p in proposals.items():
            mu_m, sd_m = flat(_weightedMean(p)), flat(_weightedStd(p))
            with np.errstate(divide='ignore', invalid='ignore'):
                shift = np.where(sel, np.abs(mu_m - mu_n) / sd_n, np.nan)
                log_ratio = np.where(sel, np.log(sd_m / sd_n), np.nan)
            worst = np.nan_to_num(np.abs(log_ratio), nan=-1.0).argmax(1)
            out[f'{m}:shift'] = np.fmax.reduce(shift, axis=1)
            # position in [ffx, sigma_rfx, sigma_eps] (padded to the model's d and q)
            out[f'{m}:shift_at'] = np.nan_to_num(shift, nan=-1.0).argmax(1)
            out[f'{m}:sd_ratio'] = np.exp(log_ratio[np.arange(B), worst])
            out[f'{m}:sd_ratio_at'] = worst
            if m in IMH_MOVES:
                out[f'{m}:moved'] = movesRan(p)
        return out

    return hook


def apiFlags(agree: dict[str, np.ndarray], imh: str | None) -> dict[str, np.ndarray]:
    """The datasets the Api warns about, for the IMH run ``imh`` and, where ``agree`` holds
    whether its moves ran, for ``{imh}+moves`` (see the module docstring)."""
    if imh is None:
        return {}
    flags = {imh: imhFlagged(agree[f'{imh}:accept'], agree[f'{imh}:ess_min'])}
    moves = f'{imh}+moves'
    if f'{moves}:moved' in agree:
        # the moves ran exactly where the run's IMH min ESS fell below IMH_ESS_WARN
        ess = np.where(agree[f'{moves}:moved'], 0.0, np.inf)
        flags[moves] = imhFlagged(agree[f'{moves}:accept'], ess)
    return flags


def renderShiftMd(
    pooled: dict[str, dict[str, np.ndarray]],
    labels: dict[str, str],
    methods: list[str],
    imh: str | None,
) -> str:
    rows = []
    for tag, agree in pooled.items():
        conv = agree['conv'].astype(bool)
        flags = apiFlags(agree, imh)
        for j, m in enumerate(methods):
            shift, ratio = agree[f'{m}:shift'][conv], agree[f'{m}:sd_ratio'][conv]
            off = (shift > SHIFT_MAX) | (np.abs(np.log(ratio)) > np.log(SD_RATIO_MAX))
            flag = flags.get(m)
            rows.append(
                [
                    labels[tag] if j == 0 else '',
                    METHOD_LABELS.get(m, m),
                    int(conv.sum()) if j == 0 else '',
                    f'{np.median(shift):.2f}',
                    f'{np.median(ratio):.2f}',
                    int(off.sum()),
                    int((off & ~flag[conv]).sum()) if flag is not None else '-',
                ]
            )
    headers = [
        'condition',
        'method',
        'n_conv',
        'median |Δmean| / SD_NUTS',
        'median SD ratio',
        'n off',
        'n off, unflagged',
    ]
    return tabulate(rows, headers=headers, tablefmt='pipe', stralign='right')


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description='MB<->NUTS agreement along the stress-search axes.')
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--axes', type=str, nargs='+', default=list(AXES), choices=list(AXES))
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--methods', type=str, nargs='*', default=None,
                        help='refinements added as extra rows (default: family preset); MB always included')
    parser.add_argument('--outdir', type=str, default=str(RESULTS_DIR))
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
    imh = next((m for m in methods if m in IMH_METHODS and m not in IMH_MOVES), None)
    # every level of a dataset stresses the same column, and the baseline holds the same datasets
    with open(DATA_DIR / f'{SIZE}-{family}-lev2' / 'config.yaml') as f:
        stress_col = np.array(yaml.safe_load(f)['stress_col'])

    for axis in cfg.axes:
        conditions = [BASELINE] + AXES[axis][family]
        if len(conditions) == 1:
            logger.info('%s: no %s axis', family, axis)
            continue
        labels = dict(conditions)
        hook = stressedShift(stress_col if axis == 'leverage' else None)
        pooled = {}
        for tag, _ in conditions:
            collected = collectCondition(cfg, family, SIZE, tag, device, methods, extra=hook)
            if collected is not None:
                pooled[tag] = collected['agree']

        rows_agree = agreementRows(pooled, labels, methods)
        agree_md = renderAgreementMd(rows_agree, dp=cfg.decimals)
        diag_md = renderDiagnosticsMd(pooled, labels, methods)
        shift_md = renderShiftMd(pooled, labels, methods, imh)
        stressed = (
            'the coefficient of the stressed predictor'
            if axis == 'leverage'
            else 'the worst active global parameter (fixed effects, random-effect SDs'
            + (', σ_ε)' if lf == 0 else ')')
        )
        md = [
            f'# Stress search, {axis} axis ({family})\n',
            f'{SIZE} datasets, NUTS (nuts2) refit per level under the stored prior. '
            'Agreement on the NUTS-converged subset (paired); flags over all datasets.\n',
            '## MB<->NUTS agreement (median ± MAD over converged datasets)\n',
            agree_md,
            '',
            '## IMH acceptance and flags (all datasets)\n',
            diag_md,
            '',
            '## Stressed parameters (converged datasets)\n',
            f'Stressed: {stressed}. Off NUTS: |Δmean| > {SHIFT_MAX:g} NUTS posterior SD or '
            f'SD ratio beyond {SD_RATIO_MAX:g}x either way. Unflagged: no Api warning for that '
            f'method (acceptance < {IMH_ACCEPT_WARN:g} or min ESS < {IMH_ESS_WARN:g} on its own '
            'IMH run, so with the random-walk moves wherever they ran); none for the raw flow.\n',
            shift_md,
            '',
        ]
        for title, table in [('agreement', agree_md), ('flags', diag_md), ('stressed', shift_md)]:
            print(f'\n=== {family} {axis}: {title} ===\n\n{table}')

        stem = f'stress_{family}_{axis}'
        (outdir / f'{stem}.md').write_text('\n'.join(md) + '\n')
        (outdir / f'{stem}.tex').write_text(renderAgreementTex(rows_agree, dp=cfg.decimals))
        np.savez(
            outdir / f'{stem}.npz',
            **{
                f'{tag}/{k}': v
                for tag, agree in pooled.items()
                for k, v in agree.items()
                if v.dtype != object
            },
        )
        logger.info('Saved tables to %s', outdir / f'{stem}.md')


if __name__ == '__main__':
    main()
