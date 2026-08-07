"""Prior-family sensitivity: does the posterior actually respond to the family encoding?

metabeta is handed the prior *family* (Normal / Student-t for β; Half-Normal /
Half-Student-t / Exponential for the scales) as a categorical context feature alongside the
scale τ.  A model that ignored that feature would still look calibrated on average, so this
script tests it directly: hold the data and every other hyperparameter fixed, overwrite one
family index across the whole test set, resample, and compare the posterior spread.

The families are ordered by their prior SD in units of τ — the no-data limit of the effect:

    β         Normal 1.000  <  Student-t(ν=5) 1.291
    σ         Half-Normal 0.603  <  Half-Student-t(ν=5) 0.875  <  Exponential 1.000

so a responsive model should widen monotonically along each ordering, by a factor between 1
(data-dominated) and the prior ratio (prior-dominated).  Reported per group as median ± MAD
over active parameter entries, in the model's standardized space (τ's own units — the
posterior is deliberately *not* rescaled by sd_y, so SDs are directly comparable to τ).

Like prior_misspec.py, the precomputed analytical MAP/EB ``stats`` are dropped and recomputed
under the assumed family, since they depend on the prior too.  Posterior samples are cached
per family setting next to the data (``variant=fam-<group>-<family>``).

Sizes are pooled; each size uses its regime-matched checkpoint (BEST_SEEDS from
scripts/build_ckpt.py).  Non-Normal likelihoods have no σ_ε, so that group is skipped.

Usage (from repo root):
    uv run python experiments/evaluation/prior_families.py --family n
    uv run python experiments/evaluation/prior_families.py --family b --sizes small --plot
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF, hasSigmaEps
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.device import setDevice
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, REPO_ROOT
from metabeta.utils.logger import setupLogging
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed
from metabeta.utils.posterior_eval import loadModel, loadOrSampleMB

# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM, _fmtMs
from real_posterior import _ms
from data_poverty import _flat
from likelihood_misspec import FAMILY_NAMES
from prior_misspec import DEFAULT_SIZES

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR

FAMILY_LABELS = {
    'normal': 'Normal',
    'student': 'Student-t',
    'halfnormal': 'Half-Normal',
    'halfstudent': 'Half-Student-t',
    'exponential': 'Exponential',
}

# (batch key holding the family index, posterior parameter it governs, ordered families)
GROUPS = [
    ('family_ffx', 'ffx', FFX_FAMILIES),
    ('family_sigma_rfx', 'sigma_rfx', SIGMA_FAMILIES),
    ('family_sigma_eps', 'sigma_eps', SIGMA_FAMILIES),
]
PARAM_LABELS = {'ffx': 'β', 'sigma_rfx': 'σ_rfx', 'sigma_eps': 'σ_ε'}


# ---------------------------------------------------------------------------
# Prior SD in units of τ — the no-data ceiling on the family effect


def _halfStudentSd(df: float) -> float:
    """SD of |T_df| for a unit-scale Student-t: sqrt(E[T²] − E|T|²)."""
    e_abs = (
        2.0
        * math.sqrt(df)
        * math.gamma((df + 1) / 2)
        / ((df - 1) * math.sqrt(math.pi) * math.gamma(df / 2))
    )
    return math.sqrt(df / (df - 2) - e_abs**2)


PRIOR_SD: dict[str, float] = {
    'normal': 1.0,
    'student': math.sqrt(STUDENT_DF / (STUDENT_DF - 2)),
    'halfnormal': math.sqrt(1.0 - 2.0 / math.pi),
    'halfstudent': _halfStudentSd(STUDENT_DF),
    'exponential': 1.0,
}


# ---------------------------------------------------------------------------
# Per-size collection


def setPriorFamily(
    batch: dict[str, torch.Tensor],
    key: str,
    index: int,
) -> dict[str, torch.Tensor]:
    """Shallow copy of ``batch`` with every dataset's ``key`` family index set to ``index``.

    ``stats`` is dropped for the same reason as in prior_misspec.perturbPrior: the analytical
    MAP/EB context is a function of the prior family, so it must be recomputed rather than
    inherited from the correct-family precomputation.
    """
    out = dict(batch)
    out.pop('stats', None)
    out[key] = torch.full_like(batch[key], index)
    return out


def _entries(
    proposal: Proposal,
    batch: dict[str, torch.Tensor],
    param_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per active entry: posterior (sd, mean) and the dataset's n, flattened.

    Draws are equal-weight (raw flow posterior, no post-hoc reweighting), so plain moments
    over the sample axis are exact.
    """
    if param_key == 'sigma_eps':
        samples = proposal.sigma_eps.unsqueeze(-1)   # (B, s, 1)
        mask = torch.ones(samples.shape[0], 1, dtype=torch.bool)
    else:
        samples = getattr(proposal, param_key)   # (B, s, d/q)
        mask = batch['mask_d' if param_key == 'ffx' else 'mask_q'].bool()
    sd = _flat(samples.std(dim=-2), mask)
    mean = _flat(samples.mean(dim=-2), mask)
    n = _flat(batch['n'].unsqueeze(-1).expand(-1, mask.shape[-1]).float(), mask)
    return sd, mean, n


def collectSize(
    cfg: argparse.Namespace,
    family: str,
    size: str,
    device: torch.device,
) -> dict | None:
    """Posterior spread per (parameter group × prior family) for one (family, size)."""
    data_id = f'{size}-{family}-sampled'
    seed = BEST_SEEDS.get((FAMILY_NAMES[family], size))
    if seed is None:
        logger.warning(
            '%s: no BEST_SEEDS checkpoint for (%s, %s) — skipping', data_id, family, size
        )
        return None
    ckpt_dir = _ckpt_dir(FAMILY_NAMES[family], size, seed)
    data_path = DATA_DIR / data_id / 'test.fit.npz'
    if not data_path.exists() or not ckpt_dir.exists():
        logger.warning('%s: data or checkpoint missing — skipping', data_id)
        return None

    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)
    max_d, max_q, lf = model_cfg.max_d, model_cfg.max_q, model_cfg.likelihood_family

    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B = len(col)
    batch = collateGrouped([col[i] for i in range(B)])
    logger.info('%s: %d datasets', data_id, B)

    out: dict[str, dict] = {'sd': {}, 'mean': {}, 'n': {}}
    for family_key, param_key, families in activeGroups(lf):
        out['sd'][param_key] = {}
        out['mean'][param_key] = {}
        for index, name in enumerate(families):
            logger.info('%s: %s = %s', data_id, family_key, name)
            batch_f = setPriorFamily(batch, family_key, index)
            proposal, _ = loadOrSampleMB(
                model,
                batch_f,
                data_path,
                ckpt_dir,
                cfg.prefix,
                cfg.n_samples,
                cfg.batch_size,
                cfg.seed,
                device,
                None,
                variant=f'fam-{param_key}-{name}',
            )
            sd, mean, n = _entries(proposal, batch_f, param_key)
            out['sd'][param_key][name] = sd
            out['mean'][param_key][name] = mean
            out['n'][param_key] = n
            del batch_f, proposal
    out['n_raw'] = batch['n'].float().numpy()
    return out


def activeGroups(lf: int) -> list[tuple[str, str, tuple[str, ...]]]:
    """Parameter groups present for a likelihood family (σ_ε is Normal-only)."""
    return [g for g in GROUPS if g[1] != 'sigma_eps' or hasSigmaEps(lf)]


def poolSizes(per_size: list[dict]) -> dict:
    """Concatenate per-entry arrays across sizes (param dims differ, so entries are flat)."""
    pooled: dict[str, dict] = {'sd': {}, 'mean': {}, 'n': {}}
    for param_key in per_size[0]['sd']:
        pooled['n'][param_key] = np.concatenate([r['n'][param_key] for r in per_size])
        for stat in ('sd', 'mean'):
            pooled[stat][param_key] = {
                name: np.concatenate([r[stat][param_key][name] for r in per_size])
                for name in per_size[0][stat][param_key]
            }
    pooled['n_raw'] = np.concatenate([r['n_raw'] for r in per_size])
    return pooled


# ---------------------------------------------------------------------------
# Tables


def spreadRows(pooled: dict, lf: int) -> list[dict]:
    """One row per (parameter group × prior family), ratios against the narrowest family."""
    rows = []
    for _, param_key, families in activeGroups(lf):
        ref_name = families[0]
        ref_sd = pooled['sd'][param_key][ref_name]
        ref_mean = pooled['mean'][param_key][ref_name]
        for j, name in enumerate(families):
            sd = pooled['sd'][param_key][name]
            mean = pooled['mean'][param_key][name]
            is_ref = j == 0
            ratio = sd / np.maximum(ref_sd, 1e-12)
            rows.append(
                {
                    'group': PARAM_LABELS[param_key],
                    'family': FAMILY_LABELS.get(name, name),
                    'first': is_ref,
                    'n_entries': len(sd),
                    'sd': _ms(sd),
                    'ratio': None if is_ref else _ms(ratio),
                    'pct_wider': None if is_ref else 100.0 * float((sd > ref_sd).mean()),
                    'dmean': None if is_ref else _ms(np.abs(mean - ref_mean)),
                    'prior_ratio': PRIOR_SD[name] / PRIOR_SD[ref_name],
                }
            )
    return rows


SPREAD_COLS = [
    ('sd', 'post. SD'),
    ('ratio', 'SD-ratio'),
    ('dmean', 'Δ|mean|'),
]


def renderSpreadMd(rows: list[dict], dp: int = 3) -> str:
    headers = ['group', 'prior family', 'entries'] + [h for _, h in SPREAD_COLS]
    headers += ['% wider', 'prior ratio']
    md = []
    for r in rows:
        lead = [r['group'], r['n_entries']] if r['first'] else ['', '']
        pct = 'ref' if r['pct_wider'] is None else f"{r['pct_wider']:.0f}"
        cells = ['ref' if r[k] is None else _fmtMs(r[k], dp) for k, _ in SPREAD_COLS]
        md.append([lead[0], r['family'], lead[1]] + cells + [pct, f"{r['prior_ratio']:.{dp}f}"])
    # numparse would re-render the pre-formatted ratio column (1.000 -> 1)
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right', disable_numparse=True)


def renderSpreadTex(rows: list[dict], dp: int = 3) -> str:
    def cell(val):
        if val is None:
            return r'\textrm{ref}'
        m, s = val
        return r'\textrm{NA}' if m != m else f'${m:.{dp}f} \\pm {s:.{dp}f}$'

    header = (
        r'\mathrm{group} & \mathrm{prior\ family} & \mathrm{entries} & \mathrm{SD} & '
        r'\mathrm{SD\text{-}ratio} & \Delta|\mathrm{mean}| & \%\,\mathrm{wider} & '
        r'\mathrm{prior\ ratio}'
    )
    lines = [
        r'\begin{tabular}{llr|ccc|rr}',
        r'    \toprule',
        f'    {header} \\\\',
        r'    \midrule',
    ]
    for i, r in enumerate(rows):
        if r['first'] and i != 0:
            lines.append(r'    \midrule')
        lead = (rf"\texttt{{{r['group']}}}", r['n_entries']) if r['first'] else ('', '')
        pct = r'\textrm{ref}' if r['pct_wider'] is None else f"{r['pct_wider']:.0f}"
        cells = ' & '.join(cell(r[k]) for k, _ in SPREAD_COLS)
        lines.append(
            rf"    {lead[0]} & \texttt{{{r['family']}}} & {lead[1]} & {cells} & "
            rf"{pct} & ${r['prior_ratio']:.{dp}f}$ \\"
        )
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def monotonicityReport(pooled: dict, lf: int) -> list[str]:
    """PASS/FAIL per consecutive family pair: does the wider prior widen the posterior?

    Uses the same statistic as the table's SD-ratio column — the median over entries of the
    *paired* ratio sd_b/sd_a — rather than a ratio of medians.  Each entry is the same
    parameter of the same dataset under two priors, so pairing is exact and the two readings
    must not be allowed to disagree in sign on near-tied families.
    """
    out = []
    for _, param_key, families in activeGroups(lf):
        order = ' < '.join(FAMILY_LABELS.get(f, f) for f in families)
        out.append(f'[{PARAM_LABELS[param_key]}] prior spread ordering: {order}')
        for a, b in zip(families, families[1:]):
            sd_a = pooled['sd'][param_key][a]
            sd_b = pooled['sd'][param_key][b]
            ratio = float(np.median(sd_b / np.maximum(sd_a, 1e-12)))
            prior_ratio = PRIOR_SD[b] / PRIOR_SD[a]
            status = 'PASS' if ratio > 1.0 else 'FAIL'
            out.append(
                f'  {FAMILY_LABELS.get(a, a)} -> {FAMILY_LABELS.get(b, b)}: '
                f'SD-ratio={ratio:.3f} (prior {prior_ratio:.3f}) [{status}]'
            )
    return out


# ---------------------------------------------------------------------------
# Plot


SIZE_RANGE = (15.0, 120.0)


def _panels(lf: int) -> list[tuple[str, str, list[str]]]:
    """(narrower family, wider family, overlaid parameter groups) — consecutive pairs."""
    panels = [(a, b, ['ffx']) for a, b in zip(FFX_FAMILIES, FFX_FAMILIES[1:])]
    keys = ['sigma_rfx'] + (['sigma_eps'] if hasSigmaEps(lf) else [])
    panels += [(a, b, keys) for a, b in zip(SIGMA_FAMILIES, SIGMA_FAMILIES[1:])]
    return panels


def _pointSizes(n: np.ndarray) -> np.ndarray:
    lo, hi = n.min(), n.max()
    if hi > lo:
        return np.interp(n, (lo, hi), SIZE_RANGE)
    return np.full_like(n, float(np.mean(SIZE_RANGE)), dtype=float)


def plotSpread(pooled: dict, lf: int, out_path: Path, delta: bool = False) -> None:
    """Posterior SD under the narrower vs the wider prior family, one panel per pair."""
    from matplotlib import pyplot as plt

    from metabeta.utils.plot import PALETTE, niceify

    colors = {'ffx': PALETTE[0], 'sigma_rfx': PALETTE[2], 'sigma_eps': PALETTE[4]}
    tex_labels = {
        'ffx': r'$\beta$',
        'sigma_rfx': r'$\sigma_\alpha$',
        'sigma_eps': r'$\sigma_\epsilon$',
    }
    panels = _panels(lf)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.7 * len(panels), 6), dpi=300)
    axes = np.atleast_1d(axes)

    for ax, (light, heavy, param_keys) in zip(axes, panels):
        xs, hs, ss, cs = [], [], [], []
        for pk in param_keys:
            xs.append(pooled['sd'][pk][light])
            hs.append(pooled['sd'][pk][heavy])
            ss.append(_pointSizes(pooled['n'][pk]))
            cs.extend([colors[pk]] * len(xs[-1]))
        x, sd_heavy, s = np.concatenate(xs), np.concatenate(hs), np.concatenate(ss)
        y = sd_heavy - x if delta else sd_heavy

        ax.scatter(x, y, s=s, c=cs, alpha=0.45, edgecolors='none')
        if delta:
            ax.axhline(0, ls='--', color='grey', lw=1, alpha=0.6)
        else:
            lo = float(min(x.min(), y.min())) * 0.95
            hi = float(max(x.max(), y.max())) * 1.05
            ax.plot([lo, hi], [lo, hi], '--', color='grey', lw=1, alpha=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        light_lbl = FAMILY_LABELS.get(light, light)
        heavy_lbl = FAMILY_LABELS.get(heavy, heavy)
        sd_ratio = float(np.median(sd_heavy / np.maximum(x, 1e-12)))
        # a fraction, not a percentage: niceify renders every stat with 3 decimals
        wider = float((sd_heavy > x).mean())
        niceify(
            ax,
            {
                'title': f'{light_lbl} vs. {heavy_lbl}',
                'title_fs': 22,
                'xlabel': f'SD({light_lbl})',
                'xlabel_fs': 20,
                'ylabel': (rf'$\Delta$ SD({heavy_lbl})' if delta else f'SD({heavy_lbl})'),
                'ylabel_fs': 20,
                'show_legend': False,
                'stats': {'SD-ratio': sd_ratio, 'frac. wider': wider},
                'stats_suffix': '',
                'stats_loc_x': 0.6,
                'stats_loc_y': 0.05,
            },
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=colors[pk],
            markersize=16,
            label=tex_labels[pk],
        )
        for pk in pooled['sd']
    ]
    n_raw = pooled['n_raw']
    levels = np.unique(np.round(np.quantile(n_raw, [0.0, 0.5, 1.0])).astype(int))
    lo, hi = n_raw.min(), n_raw.max()
    size_handles = [
        plt.scatter(
            [],
            [],
            s=float(np.interp(float(nl), (lo, hi), SIZE_RANGE)) if hi > lo else 60.0,
            c='grey',
            alpha=0.5,
            label=f'{nl}',
        )
        for nl in levels
    ]
    leg = axes[-1].legend(
        handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.75), fontsize=20
    )
    axes[-1].add_artist(leg)
    axes[-1].legend(
        handles=size_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.30),
        fontsize=20,
        title='n (total)',
        title_fontsize=20,
    )

    fig.suptitle('Posterior spread under the prior family', fontsize=26, x=0.42, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info('Saved plot to %s', out_path)


# ---------------------------------------------------------------------------
# CLI / main


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description='Prior-family sensitivity of the MB posterior.')
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true', help='also write the scatter figure')
    parser.add_argument('--delta', action='store_true', help='plot ΔSD instead of absolute SD')
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

    sizes = [s for s in cfg.sizes if BEST_SEEDS.get((FAMILY_NAMES[family], s)) is not None]
    per_size = [r for r in (collectSize(cfg, family, s, device) for s in sizes) if r is not None]
    if not per_size:
        logger.error('No sizes evaluated.')
        return

    pooled = poolSizes(per_size)
    rows = spreadRows(pooled, lf)
    spread_md = renderSpreadMd(rows, dp=cfg.decimals)
    report = monotonicityReport(pooled, lf)

    print('\n=== Posterior spread by prior family (median ± MAD over active entries) ===\n')
    print(spread_md)
    print()
    print('\n'.join(report))

    stem = f'prior_families_{family}'
    md = [
        f'# Prior-family sensitivity ({family})\n',
        f'Sizes: {", ".join(sizes)}. Data and all other hyperparameters held fixed; one family '
        'index overwritten across the test set per row. Posterior moments in standardized '
        "space (τ's units), draws equal-weight. Ratios are against the narrowest-prior family "
        'of each group; ``prior ratio`` is the no-data ceiling implied by the prior SDs.\n',
        spread_md,
        '',
        '## Monotonicity\n',
        '```',
        *report,
        '```',
        '',
    ]
    (outdir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    (outdir / f'{stem}.tex').write_text(renderSpreadTex(rows, dp=cfg.decimals))
    logger.info('Saved tables to %s', outdir / f'{stem}.md')

    if cfg.plot:
        plotSpread(pooled, lf, outdir / f'{stem}.png', delta=cfg.delta)


if __name__ == '__main__':
    main()
