"""Audit NUTS convergence diagnostics of the pre-computed baseline fits.

Produces the numbers cited in the paper appendix paragraph "NUTS convergence
diagnostics" (metabeta-paper/appendices/pymc.tex): per-benchmark divergence
prevalence, R-hat / ESS / tree-depth summaries, convergence rates under the
strict and liberal criteria of ``nutsConvergeMask``, the sigma_eps driver
analysis (Normal only, where the true generative sigma_eps is known), and
Spearman correlations between sampler health and the NUTS LOO-NLL.

Only the small diagnostic arrays of each test.fit.npz are read (never the
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
import torch
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from tabulate import tabulate

from metabeta.utils.evaluation import EvaluationSummary, nutsConvergeMask
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR

FAMILIES = {'n': 'Normal', 'b': 'Bernoulli', 'p': 'Poisson'}
SIZES = ['small', 'medium', 'large', 'huge']
VARIANTS = ['sampled', 'real']

DIAG_KEYS = [
    'nuts_divergences',
    'nuts_rhat',
    'nuts_ess',
    'nuts_ess_tail',
    'nuts_max_treedepth',
    'nuts_draws',
]
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


def loadDiagnostics(path: Path) -> dict[str, np.ndarray] | None:
    """Read only the diagnostic members of a reintegrated test.fit.npz."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as f:
        if 'nuts_divergences' not in f.files:
            return None
        return {k: f[k] for k in DIAG_KEYS + EXTRA_KEYS if k in f.files}


def _paramStat(arr: np.ndarray, fn) -> np.ndarray:
    """Per-dataset statistic over parameters, treating padded entries (<= 0) as missing."""
    a = arr.astype(np.float64).copy()
    a[a <= 0] = np.nan
    return fn(a, axis=-1)


def perDataset(diag: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-dataset diagnostic arrays shared by the table and the driver analysis."""
    out = {
        'total_div': diag['nuts_divergences'].sum(-1),
        'max_rhat': _paramStat(diag['nuts_rhat'], np.nanmax),
        'min_ess': _paramStat(diag['nuts_ess'], np.nanmin),
        'td_sat': diag['nuts_max_treedepth'].mean(-1),
    }
    torch_batch = {k: torch.as_tensor(v) for k, v in diag.items() if k in DIAG_KEYS}
    # the reintegrated batch stores nuts_draws per dataset; the mask expects a scalar
    torch_batch['nuts_draws'] = torch_batch['nuts_draws'].reshape(-1)[0]
    for mode in ('strict', 'liberal'):
        out[f'conv_{mode}'] = nutsConvergeMask(torch_batch, mode=mode)
    return out


def benchmarkRow(data_id: str, per: dict[str, np.ndarray]) -> dict:
    total_div, max_rhat, td_sat = per['total_div'], per['max_rhat'], per['td_sat']
    b = len(total_div)
    affected = total_div[total_div > 0]
    return {
        'benchmark': data_id,
        'B': b,
        'pct_any_div': 100.0 * (total_div > 0).mean(),
        'total_div': int(total_div.sum()),
        'med_div_affected': float(np.median(affected)) if len(affected) else 0.0,
        'pct_rhat': 100.0 * np.mean(max_rhat > 1.01),
        'max_rhat': float(np.nanmax(max_rhat)),
        'pct_tree': 100.0 * np.mean(td_sat > 0.05),
        'pct_conv_strict': 100.0 * per['conv_strict'].mean(),
        'pct_conv_liberal': 100.0 * per['conv_liberal'].mean(),
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
    data_id: str, diag: dict[str, np.ndarray], per: dict[str, np.ndarray], data_dir: Path
) -> dict | None:
    """sigma_eps / geometry driver analysis for one Normal sampled benchmark."""
    if 'sigma_eps' not in diag:
        return None
    total_div, max_rhat, min_ess = per['total_div'], per['max_rhat'], per['min_ess']
    sigma_std = diag['sigma_eps'] / diag['sd_y']
    low = sigma_std < SIGMA_EPS_THRESHOLD
    mq = diag['m'].astype(np.float64) * diag['q'].astype(np.float64)
    bad_rhat = max_rhat > 1.01
    loo = loadLooNll(data_dir, len(total_div))
    return {
        'benchmark': data_id,
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


def variantRow(family: str, size: str, per_s: dict, per_r: dict) -> dict:
    """Sampled-vs-real contrast for one (family, size) pair.

    Divergence prevalence is compared with Fisher's exact test on the ≥1-divergence
    counts; the per-dataset divergence-count distributions with a two-sided
    Mann-Whitney U; the R-hat violation shares again with Fisher's exact test.
    """
    div_s, div_r = per_s['total_div'], per_r['total_div']
    any_s, any_r = div_s > 0, div_r > 0
    bad_s, bad_r = per_s['max_rhat'] > 1.01, per_r['max_rhat'] > 1.01

    def fisher(a: np.ndarray, b: np.ndarray) -> float:
        table = [[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]
        return float(fisher_exact(table)[1])

    p_any = fisher(any_s, any_r)
    p_mw = float(mannwhitneyu(div_s, div_r, alternative='two-sided')[1])
    p_rhat = fisher(bad_s, bad_r)
    return {
        'pair': f'{size}-{family}',
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
    ('B', 'B', '{}'),
    ('total_div', 'divg.', '{}'),
    ('pct_any_div', '% ≥1 divg.', '{:.0f}'),
    ('med_div_affected', 'med. divg.|>0', '{:.0f}'),
    ('pct_rhat', '% R̂>1.01', '{:.1f}'),
    ('max_rhat', 'max R̂', '{:.2f}'),
    ('pct_tree', '% tree-sat', '{:.1f}'),
    ('pct_conv_strict', '% conv (strict)', '{:.0f}'),
    ('pct_conv_liberal', '% conv (liberal)', '{:.0f}'),
]

VARIANT_COLS = [
    ('pair', 'benchmark pair', '{}'),
    ('pct_any_s', '% ≥1 divg. (sampled)', '{:.0f}'),
    ('pct_any_r', '% ≥1 divg. (real)', '{:.0f}'),
    ('p_any', 'p (Fisher)', '{}'),
    ('p_mw', 'p (MW, counts)', '{}'),
    ('pct_rhat_s', '% R̂>1.01 (sampled)', '{:.1f}'),
    ('pct_rhat_r', '% R̂>1.01 (real)', '{:.1f}'),
    ('p_rhat', 'p (Fisher)', '{}'),
]

DRIVER_COLS = [
    ('benchmark', 'benchmark', '{}'),
    ('pct_low_sigma', f'% σ̃ε<{SIGMA_EPS_THRESHOLD}', '{:.0f}'),
    ('pct_div_low', '% divg.|σ̃ε low', '{:.0f}'),
    ('pct_div_rest', '% divg.|rest', '{:.0f}'),
    ('rho_sigma_div', 'ρ(σ̃ε, divg.)', '{}'),
    ('med_mq_bad', 'med. m·q|R̂>1.01', '{:.0f}'),
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
        r'    $\mathrm{benchmark}$ & $B$ & $\mathrm{divg.}$ & $\%\,{\ge}1\,\mathrm{divg.}$ & '
        r'$\%\,\hat{R}_\mathrm{max}{>}1.01$ & $\hat{R}_\mathrm{worst}$ & '
        r'$\%\,\mathrm{tree\text{-}sat.}$ & $\%\,\mathrm{conv.\,(strict)}$ & '
        r'$\%\,\mathrm{conv.\,(liberal)}$ \\'
    )
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'    \toprule', header, r'    \midrule']
    prev_family = None
    for r in rows:
        family = r['benchmark'].split('-')[1]
        if prev_family is not None and family != prev_family:
            lines.append(r'    \midrule')
        prev_family = family
        div = f"{r['total_div']:,}".replace(',', r'{,}')
        lines.append(
            rf"    \texttt{{{r['benchmark']}}} & ${r['B']}$ & ${div}$ & "
            rf"${r['pct_any_div']:.0f}$ & ${r['pct_rhat']:.1f}$ & ${r['max_rhat']:.2f}$ & "
            rf"${r['pct_tree']:.1f}$ & ${r['pct_conv_strict']:.0f}$ & "
            rf"${r['pct_conv_liberal']:.0f}$ \\"
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
            per_variant: dict[str, dict[str, np.ndarray]] = {}
            for variant in args.variants:
                data_id = f'{size}-{family}-{variant}'
                diag = loadDiagnostics(DATA_DIR / data_id / 'test.fit.npz')
                if diag is None:
                    continue
                per = perDataset(diag)
                per_variant[variant] = per
                bench_rows.append(benchmarkRow(data_id, per))
                if family == 'n' and variant == 'sampled':
                    row = driverRow(data_id, diag, per, DATA_DIR / data_id)
                    if row is not None:
                        driver_rows.append(row)
            if 'sampled' in per_variant and 'real' in per_variant:
                variant_rows.append(
                    variantRow(family, size, per_variant['sampled'], per_variant['real'])
                )

    if not bench_rows:
        raise ValueError('no reintegrated test.fit.npz with NUTS diagnostics found')

    bench_md = renderMd(bench_rows, BENCH_COLS)
    print('\n=== NUTS convergence audit (test.fit.npz, 4 chains x 1000 draws) ===\n')
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
        'Diagnostics from the reintegrated `test.fit.npz` baselines '
        '(4 chains, 2,000 tuning steps, 1,000 draws, target accept 0.8).\n'
        'Tree-sat: fraction of datasets whose chains saturate max tree depth on >5% of draws.\n'
        'Convergence criteria: see `nutsConvergeMask` in `metabeta/utils/evaluation.py`.\n\n'
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
            'LOO-NLL is the NUTS posterior predictive LOO-NLL from `summary_test_nuts.pt`. '
            'Spearman stars: *p<.05, **p<.01, ***p<.001.\n\n'
            f'{driver_md}\n'
        )
    (outdir / 'nuts_divergences.md').write_text(md)
    (outdir / 'nuts_divergences.tex').write_text(renderBenchmarkTex(bench_rows))
    print(f'\nSaved {outdir / "nuts_divergences.md"} and .tex')


if __name__ == '__main__':
    main()
