"""Audit NUTS convergence diagnostics of the pre-computed baseline fits.

Produces the numbers cited in the paper appendix paragraph "NUTS convergence
diagnostics" (metabeta-paper/appendices/pymc.tex): per benchmark and budget
level (``nuts0`` < ``nuts1`` < ``nuts2``) the divergence prevalence, R-hat /
tree-depth summaries, the fraction converged under the single ``nutsConverged``
criterion and the share failing each of its checks (R-hat, bulk ESS, tail ESS,
divergence rate), the sigma_eps driver analysis (Normal only, where the true
generative sigma_eps is known), and Spearman correlations between sampler health
and the NUTS LOO-NLL.

Only the small diagnostic arrays of each test.nuts{k}.npz are read (never the
posterior sample tensors); per-dataset LOO-NLL comes from the cached
full-split NUTS evaluation summary (summary_test_nuts.pt) written by
evaluate.py / ablation.py, where available.

Run from repo root:
    uv run python experiments/evaluation/nuts_divergences.py
    uv run python experiments/evaluation/nuts_divergences.py --families n --variants sampled
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from tabulate import tabulate

from metabeta.utils.evaluation import (
    ESS_MIN,
    NUTS_DIAG_KEYS,
    RHAT_MAX,
    EvaluationSummary,
    nutsChecks,
)
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR
from metabeta.utils.fits import fitPath, loadFits

FAMILIES = {'n': 'Normal', 'b': 'Bernoulli', 'p': 'Poisson'}
SIZES = ['small', 'medium', 'large', 'huge']
VARIANTS = ['sampled', 'real']
LEVELS = ('nuts0', 'nuts1', 'nuts2')  # the NUTS budget ladder, one fit file each

DIAG_KEYS = (*NUTS_DIAG_KEYS, 'max_treedepth')
EXTRA_KEYS = ['sigma_eps', 'sd_y', 'm', 'q']

SIGMA_EPS_THRESHOLD = 0.10   # standardized sigma_eps below which NUTS struggles


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit NUTS convergence diagnostics across benchmark fits.')
    parser.add_argument('--families', type=str, nargs='+', default=list(FAMILIES),
                        choices=list(FAMILIES), help='Likelihood families (default: all).')
    parser.add_argument('--sizes', type=str, nargs='+', default=SIZES,
                        choices=SIZES, help='Size classes (default: all).')
    parser.add_argument('--variants', type=str, nargs='+', default=VARIANTS,
                        choices=VARIANTS, help='Benchmark variants (default: sampled and real).')
    parser.add_argument('--outdir', type=str, default=str(RESULTS_DIR))
    return parser.parse_args()
# fmt: on


def loadDiagnostics(path: Path, tag: str) -> dict[str, np.ndarray] | None:
    """Read only the diagnostic members of test.{tag}.npz (plus EXTRA_KEYS from test.npz)."""
    if not (path.exists() and fitPath(path, tag).exists()):
        return None
    diag = loadFits(path, tag, keys=[f'{tag}_{k}' for k in DIAG_KEYS])
    with np.load(path, allow_pickle=True) as f:
        diag.update({k: f[k] for k in EXTRA_KEYS if k in f.files})
    return diag


def _paramStat(arr: np.ndarray, fn) -> np.ndarray:
    """Per-dataset statistic over parameters, treating padded entries (<= 0) as missing."""
    a = arr.astype(np.float64).copy()
    a[a <= 0] = np.nan
    return fn(a, axis=-1)


def perDataset(diag: dict[str, np.ndarray], tag: str) -> dict[str, np.ndarray]:
    """Per-dataset diagnostics, prefix-free, plus the four checks of ``nutsConverged``.

    ``fail_*`` are the complements of the individual checks (a dataset converged iff none
    fails), so the table can attribute every non-converged run to its reasons.
    """
    checks = nutsChecks(diag, tag)
    return {
        'total_div': diag[f'{tag}_divergences'].sum(-1),
        'max_rhat': _paramStat(diag[f'{tag}_rhat'], np.nanmax),
        'min_ess': _paramStat(diag[f'{tag}_ess'], np.nanmin),
        'td_sat': diag[f'{tag}_max_treedepth'].mean(-1),
        **{f'fail_{k}': ~ok for k, ok in checks.items()},
        'conv': np.logical_and.reduce(list(checks.values())),
    }


def benchmarkRow(data_id: str, tag: str, per: dict[str, np.ndarray]) -> dict:
    total_div, max_rhat, td_sat = per['total_div'], per['max_rhat'], per['td_sat']
    b = len(total_div)
    affected = total_div[total_div > 0]
    return {
        'benchmark': data_id,
        'level': tag,
        'B': b,
        'pct_any_div': 100.0 * (total_div > 0).mean(),
        'total_div': int(total_div.sum()),
        'med_div_affected': float(np.median(affected)) if len(affected) else 0.0,
        'max_rhat': float(np.nanmax(max_rhat)),
        'pct_tree': 100.0 * np.mean(td_sat > 0.05),
        'pct_conv': 100.0 * per['conv'].mean(),
        'pct_fail_rhat': 100.0 * per['fail_rhat'].mean(),
        'pct_fail_ess': 100.0 * per['fail_ess'].mean(),
        'pct_fail_ess_tail': 100.0 * per['fail_ess_tail'].mean(),
        'pct_fail_div': 100.0 * per['fail_divergences'].mean(),
    }


def loadLooNll(data_dir: Path, b: int) -> np.ndarray | None:
    """Per-dataset NUTS LOO-NLL from the cached full-split summary, if it matches."""
    path = data_dir / 'summary_test_nuts.pt'
    if not path.exists():
        return None
    try:
        summary = EvaluationSummary.load(path)
    except (KeyError, ValueError, RuntimeError):
        return None
    loo = summary.per_dataset.loo_nll
    if loo is None or loo.shape[0] != b:
        return None
    return loo.double().numpy()


def _rho(x: np.ndarray, y: np.ndarray) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 'NA'
    rho, p = spearmanr(x[mask], y[mask])
    star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
    return f'{rho:+.2f}{star}'


def driverRow(
    data_id: str,
    tag: str,
    diag: dict[str, np.ndarray],
    per: dict[str, np.ndarray],
    data_dir: Path,
) -> dict | None:
    """sigma_eps / geometry driver analysis for one Normal sampled benchmark and level."""
    if 'sigma_eps' not in diag:
        return None
    total_div, min_ess = per['total_div'], per['min_ess']
    sigma_std = diag['sigma_eps'] / diag['sd_y']
    low = sigma_std < SIGMA_EPS_THRESHOLD
    mq = diag['m'].astype(np.float64) * diag['q'].astype(np.float64)
    bad_rhat = per['fail_rhat']
    loo = loadLooNll(data_dir, len(total_div))
    return {
        'benchmark': data_id,
        'level': tag,
        'pct_low_sigma': 100.0 * low.mean(),
        'pct_div_low': 100.0 * (total_div[low] > 0).mean() if low.any() else np.nan,
        'pct_div_rest': 100.0 * (total_div[~low] > 0).mean(),
        'rho_sigma_div': _rho(sigma_std, total_div),
        'med_mq_bad': float(np.median(mq[bad_rhat])) if bad_rhat.any() else np.nan,
        'med_mq_ok': float(np.median(mq[~bad_rhat])),
        'rho_div_loo': _rho(total_div, loo) if loo is not None else 'NA',
        'rho_ess_loo': _rho(min_ess, loo) if loo is not None else 'NA',
    }


def _pStar(p: float) -> str:
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''


def variantRow(family: str, size: str, tag: str, per_s: dict, per_r: dict) -> dict:
    """Sampled-vs-real contrast for one (family, size) pair at one budget level.

    Divergence prevalence is compared with Fisher's exact test on the ≥1-divergence
    counts; the per-dataset divergence-count distributions with a two-sided
    Mann-Whitney U; the R-hat violation shares again with Fisher's exact test.
    """
    div_s, div_r = per_s['total_div'], per_r['total_div']
    any_s, any_r = div_s > 0, div_r > 0
    bad_s, bad_r = per_s['fail_rhat'], per_r['fail_rhat']

    def fisher(a: np.ndarray, b: np.ndarray) -> float:
        table = [[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]
        return float(fisher_exact(table)[1])

    p_any = fisher(any_s, any_r)
    p_mw = float(mannwhitneyu(div_s, div_r, alternative='two-sided')[1])
    p_rhat = fisher(bad_s, bad_r)
    return {
        'pair': f'{size}-{family}',
        'level': tag,
        'pct_any_s': 100.0 * any_s.mean(),
        'pct_any_r': 100.0 * any_r.mean(),
        'p_any': f'{p_any:.3g}{_pStar(p_any)}',
        'p_mw': f'{p_mw:.3g}{_pStar(p_mw)}',
        'pct_rhat_s': 100.0 * bad_s.mean(),
        'pct_rhat_r': 100.0 * bad_r.mean(),
        'p_rhat': f'{p_rhat:.3g}{_pStar(p_rhat)}',
    }


# ---------------------------------------------------------------------------
# Rendering


BENCH_COLS = [
    ('benchmark', 'benchmark', '{}'),
    ('level', 'level', '{}'),
    ('B', 'B', '{}'),
    ('total_div', 'divg.', '{}'),
    ('pct_any_div', '% ≥1 divg.', '{:.0f}'),
    ('med_div_affected', 'med. divg.|>0', '{:.0f}'),
    ('max_rhat', 'max R̂', '{:.2f}'),
    ('pct_tree', '% tree-sat', '{:.1f}'),
    ('pct_conv', '% conv', '{:.0f}'),
    ('pct_fail_rhat', f'% R̂>{RHAT_MAX}', '{:.1f}'),
    ('pct_fail_ess', f'% ESS<{ESS_MIN}', '{:.1f}'),
    ('pct_fail_ess_tail', f'% ESS-tail<{ESS_MIN}', '{:.1f}'),
    ('pct_fail_div', '% divg. rate>cap', '{:.1f}'),
]

VARIANT_COLS = [
    ('pair', 'benchmark pair', '{}'),
    ('level', 'level', '{}'),
    ('pct_any_s', '% ≥1 divg. (sampled)', '{:.0f}'),
    ('pct_any_r', '% ≥1 divg. (real)', '{:.0f}'),
    ('p_any', 'p (Fisher)', '{}'),
    ('p_mw', 'p (MW, counts)', '{}'),
    ('pct_rhat_s', f'% R̂>{RHAT_MAX} (sampled)', '{:.1f}'),
    ('pct_rhat_r', f'% R̂>{RHAT_MAX} (real)', '{:.1f}'),
    ('p_rhat', 'p (Fisher)', '{}'),
]

DRIVER_COLS = [
    ('benchmark', 'benchmark', '{}'),
    ('level', 'level', '{}'),
    ('pct_low_sigma', f'% σ̃ε<{SIGMA_EPS_THRESHOLD}', '{:.0f}'),
    ('pct_div_low', '% divg.|σ̃ε low', '{:.0f}'),
    ('pct_div_rest', '% divg.|rest', '{:.0f}'),
    ('rho_sigma_div', 'ρ(σ̃ε, divg.)', '{}'),
    ('med_mq_bad', f'med. m·q|R̂>{RHAT_MAX}', '{:.0f}'),
    ('med_mq_ok', 'med. m·q|rest', '{:.0f}'),
    ('rho_div_loo', 'ρ(divg., LOO-NLL)', '{}'),
    ('rho_ess_loo', 'ρ(min ESS, LOO-NLL)', '{}'),
]


def _fmt(row: dict, cols: list[tuple[str, str, str]]) -> list[str]:
    out = []
    for key, _, fmt in cols:
        val = row[key]
        out.append('NA' if isinstance(val, float) and np.isnan(val) else fmt.format(val))
    return out


def renderMd(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    return tabulate(
        [_fmt(r, cols) for r in rows],
        headers=[h for _, h, _ in cols],
        tablefmt='pipe',
        stralign='right',
    )


def renderBenchmarkTex(rows: list[dict]) -> str:
    """LaTeX table of the per-benchmark audit, styled like the other paper tables."""
    header = (
        r'    $\mathrm{benchmark}$ & $\mathrm{level}$ & $B$ & $\mathrm{divg.}$ & '
        r'$\%\,{\ge}1\,\mathrm{divg.}$ & $\hat{R}_\mathrm{worst}$ & '
        r'$\%\,\mathrm{tree\text{-}sat.}$ & $\%\,\mathrm{conv.}$ & '
        rf'$\%\,\hat{{R}}{{>}}{RHAT_MAX}$ & $\%\,\mathrm{{ESS}}{{<}}{ESS_MIN}$ & '
        rf'$\%\,\mathrm{{ESS_{{tail}}}}{{<}}{ESS_MIN}$ & $\%\,\mathrm{{divg.\,rate}}$ \\'
    )
    lines = [r'\begin{tabular}{llrrrrrrrrrr}', r'    \toprule', header, r'    \midrule']
    prev_family = None
    for r in rows:
        family = r['benchmark'].split('-')[1]
        if prev_family is not None and family != prev_family:
            lines.append(r'    \midrule')
        prev_family = family
        div = f"{r['total_div']:,}".replace(',', r'{,}')
        lines.append(
            rf"    \texttt{{{r['benchmark']}}} & \texttt{{{r['level']}}} & ${r['B']}$ & ${div}$ & "
            rf"${r['pct_any_div']:.0f}$ & ${r['max_rhat']:.2f}$ & ${r['pct_tree']:.1f}$ & "
            rf"${r['pct_conv']:.0f}$ & ${r['pct_fail_rhat']:.1f}$ & ${r['pct_fail_ess']:.1f}$ & "
            rf"${r['pct_fail_ess_tail']:.1f}$ & ${r['pct_fail_div']:.1f}$ \\"
        )
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def main() -> None:
    args = setup()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bench_rows, driver_rows, variant_rows = [], [], []
    for family in args.families:
        for size in args.sizes:
            per_variant: dict[tuple[str, str], dict[str, np.ndarray]] = {}
            for variant in args.variants:
                data_id = f'{size}-{family}-{variant}'
                for tag in LEVELS:
                    diag = loadDiagnostics(DATA_DIR / data_id / 'test.npz', tag)
                    if diag is None:
                        continue
                    per = perDataset(diag, tag)
                    per_variant[variant, tag] = per
                    bench_rows.append(benchmarkRow(data_id, tag, per))
                    if family == 'n' and variant == 'sampled':
                        row = driverRow(data_id, tag, diag, per, DATA_DIR / data_id)
                        if row is not None:
                            driver_rows.append(row)
            for tag in LEVELS:
                if ('sampled', tag) in per_variant and ('real', tag) in per_variant:
                    variant_rows.append(
                        variantRow(
                            family, size, tag, per_variant['sampled', tag], per_variant['real', tag]
                        )
                    )

    if not bench_rows:
        raise ValueError('no test.nuts{0,1,2}.npz fit files found')

    bench_md = renderMd(bench_rows, BENCH_COLS)
    print('\n=== NUTS convergence audit (test.nuts{0,1,2}.npz, per budget level) ===\n')
    print(bench_md)

    variant_md = ''
    if variant_rows:
        variant_md = renderMd(variant_rows, VARIANT_COLS)
        print('\n=== Sampled (oracle) vs real fits ===\n')
        print(variant_md)

    driver_md = ''
    if driver_rows:
        driver_md = renderMd(driver_rows, DRIVER_COLS)
        print('\n=== Divergence drivers (Normal sampled; σ̃ε = σε/sd(y)) ===\n')
        print(driver_md)
        print('\nSpearman stars: *p<.05, **p<.01, ***p<.001')

    md = (
        '# NUTS convergence audit\n\n'
        'Diagnostics from the `test.nuts{0,1,2}.npz` fits, one row per budget level of the '
        'NUTS ladder (4 chains each; budgets in `NUTS_LEVELS`, `metabeta/simulation/fit.py`).\n'
        'Tree-sat: fraction of datasets whose chains saturate max tree depth on >5% of draws.\n'
        'Convergence: the single `nutsConverged` criterion (`metabeta/utils/evaluation.py`); '
        'the `% R̂`, `% ESS`, `% ESS-tail` and `% divg. rate` columns are the shares failing '
        'each of its checks (a dataset can fail several).\n\n'
        f'{bench_md}\n'
    )
    if variant_md:
        md += (
            '\n## Sampled (oracle) vs real fits\n\n'
            'Divergence prevalence (Fisher exact on ≥1-divergence counts), per-dataset '
            'divergence-count distributions (two-sided Mann-Whitney U), and R̂ violation '
            'shares (Fisher exact). Stars: *p<.05, **p<.01, ***p<.001.\n\n'
            f'{variant_md}\n'
        )
    if driver_md:
        md += (
            '\n## Divergence drivers (Normal sampled)\n\n'
            f'σ̃ε is the generative noise scale in standardized space (σε/sd(y)); '
            f'"% divg." columns give the share of datasets with ≥1 divergent transition.\n'
            'LOO-NLL is the composite-NUTS posterior predictive LOO-NLL from `summary_test_nuts.pt`. '
            'Spearman stars: *p<.05, **p<.01, ***p<.001.\n\n'
            f'{driver_md}\n'
        )
    (outdir / 'nuts_divergences.md').write_text(md)
    (outdir / 'nuts_divergences.tex').write_text(renderBenchmarkTex(bench_rows))
    print(f'\nSaved {outdir / "nuts_divergences.md"} and .tex')


if __name__ == '__main__':
    main()
