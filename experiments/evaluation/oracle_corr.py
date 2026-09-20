"""
Oracle evaluation of the random-effect correlations only.

The oracle tables (experiments/evaluation/oracle_posterior.py) average recovery and calibration
over fixed effects, scales, random effects and the residual scale, and exempt the correlation
parameters (paper: app:met). This script reports those exempted parameters on their own:
r / NRMSE / ECE / EACE over the lower-triangle correlation pairs, restricted to the datasets
whose ground-truth correlation matrix is not the identity (q >= 2 and eta_rfx > 0) and — as
in the oracle tables — to the NUTS-converged subset.

Efficiency: reuses the cached MB samples and post-hoc refinements written by oracle_posterior.py
(same cache keys), streams one reference method's fit tensors at a time, and computes only the
parameter-recovery block (point estimates, credible intervals, coverage errors); the predictive
/ LOO block, which dominates getSummary's runtime, is skipped because no column needs it.

Usage (from repo root):
    uv run python experiments/evaluation/oracle_corr.py --checkpoint PATH --data_id huge-n-sampled
    uv run python experiments/evaluation/oracle_corr.py --checkpoint PATH --data_id huge-n-sampled --plot
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate

# sibling experiment script (this directory is sys.path[0] at run time)
from oracle_posterior import (
    STATS,
    _capFull,
    _fmtMd,
    _fmtTex,
    fitExcludePrefixes,
    loadRegimeBatch,
    methodFitBatch,
    nutsConvergeMaskFromNpz,
)
from metabeta.evaluation.intervals import getCoverageErrors, getCoverages, getCredibleIntervals
from metabeta.evaluation.point import getCorrelation, getPointEstimates, getRMSE
from metabeta.evaluation.summary import EST_TYPE, _averageOverAlpha
from metabeta.plotting.recovery import _plotRecovery
from metabeta.utils.dataloader import subsetBatch
from metabeta.utils.device import setDevice
from metabeta.utils.evaluation import subsetProposal
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR
from metabeta.utils.logger import setupLogging
from metabeta.utils.plot import DPI, legendProxy, paramColors, savePlot
from metabeta.utils.posterior_eval import (
    SUPPORTED_METHODS,
    fit2proposal,
    fitBatchMask,
    loadModel,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
    validMethods,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import corrToLower
from metabeta.utils.results import Proposal, getCorrRfxNames
from metabeta.utils.sampling import setSeed

logger = logging.getLogger(__name__)

METRICS = ['r', 'NRMSE', 'ECE', 'EACE']
# All four columns are per-pair aggregates over the test set, so their spread runs over the
# correlation pairs (mean ± std, as the oracle tables do for parameter dimensions).
PRIMARY = 'mean ± std'


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Oracle evaluation of the random-effect correlations on one sampled test set',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_id',    type=str, required=True,
                        help='Single sampled data id to evaluate, e.g. huge-n-sampled.')
    parser.add_argument('--prefix',     type=str, default='latest')
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--n_samples',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--outdir',     type=str, default=str(RESULTS_DIR))
    parser.add_argument('--verbosity',  type=int, default=1)
    parser.add_argument('--decimals',   type=int, default=2)
    parser.add_argument('--rescale',    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup',     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--convergence_mode', type=str, default='liberal',
                        choices=['liberal', 'strict'])
    parser.add_argument('--methods',    type=str, nargs='*', default=None,
                        choices=list(SUPPORTED_METHODS),
                        help='Post-hoc refinements on top of raw MB (default: family preset).')
    parser.add_argument('--plot',       action=argparse.BooleanOptionalAction, default=False,
                        help='Also save a recovery scatter of the correlation pairs, one panel '
                             'per method in --plot_methods.')
    parser.add_argument('--plot_methods', type=str, nargs='*', default=['MB', 'NUTS', 'ADVI'],
                        help='Row labels to plot; "MB" resolves to the refined row when a '
                             'refinement ran (the paper\'s MB), "MB0" to the raw flow.')
    # fmt: on
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics


def corrPairMask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, d_corr) mask of correlation pairs whose two dimensions are both active."""
    mask_q = batch['mask_q']
    q = mask_q.shape[-1]
    if q < 2:
        return mask_q.new_zeros(mask_q.shape[0], 0, dtype=torch.bool)
    return torch.stack([mask_q[:, i] & mask_q[:, j] for i in range(q) for j in range(i)], dim=-1)


def correlatedMask(batch: dict[str, torch.Tensor]) -> np.ndarray:
    """Datasets whose true correlation matrix is not the identity (q >= 2 and eta_rfx > 0)."""
    q_active = batch['mask_q'].sum(-1)
    return ((q_active >= 2) & (batch['eta_rfx'] > 0)).numpy()


def corrMetrics(proposal: Proposal, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Recovery / calibration of the correlation pairs only (no predictive block).

    Mirrors the parameter block of getSummary (point estimates → r / NRMSE, credible intervals
    → coverage errors → ECE / EACE) and keeps only the ``corr_rfx`` entries, restricted to the
    pairs that are active in at least one dataset.
    """
    proposal.to('cpu')
    est = getPointEstimates(proposal, EST_TYPE)
    if 'corr_rfx' not in est:
        raise ValueError('proposal carries no correlation samples')
    nrmse = getRMSE(est, batch, normalize=True)['corr_rfx']
    corr = getCorrelation(est, batch)['corr_rfx']
    ci_dicts = getCredibleIntervals(proposal)
    cvrg = getCoverages(ci_dicts, batch)
    err = getCoverageErrors(cvrg, log_ratio=False)
    ece = _averageOverAlpha(err)['corr_rfx']
    eace = _averageOverAlpha(err, absolute=True)['corr_rfx']

    pair_mask = corrPairMask(batch)
    active = pair_mask.any(0)
    return {
        'r': torch.as_tensor(corr)[active].float(),
        'NRMSE': nrmse[active].float(),
        'ECE': ece[active].float(),
        'EACE': eace[active].float(),
        # for the recovery scatter
        'targets': corrToLower(batch['corr_rfx']),
        'estimates': est['corr_rfx'],
        'pair_mask': pair_mask,
    }


def buildRow(label: str, regime: str, metrics: dict[str, torch.Tensor]) -> dict:
    row: dict = {'regime': regime, 'method': label, 'stats': {}, 'plot': metrics}
    for name, fn in STATS.items():
        row['stats'][name] = {k: fn(metrics[k]) for k in METRICS}
    row.update({k: row['stats'][PRIMARY][k] for k in METRICS})
    return row


# ---------------------------------------------------------------------------
# Regime evaluation


def evaluateRegime(
    model,
    data_path: Path,
    base_path: Path,
    max_d: int,
    max_q: int,
    lf: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    regime: str,
    ckpt_dir: Path,
    prefix: str,
    seed: int,
    methods: list[str],
    rescale: bool = True,
    convergence_mode: str = 'liberal',
    warmup: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """Returns (rows, counts); rows carry the per-pair metrics of every method on the
    correlated ∩ NUTS-converged subset, counts the subset sizes for the table caption."""
    logger.info('\n--- Regime: %s (correlation parameters) ---', regime)

    data_batch, n_total, n_kept, cap_mask = loadRegimeBatch(base_path, max_d, max_q)
    logger.info('  Capacity filter: %d / %d (d≤%d, q≤%d)', n_kept, n_total, max_d, max_q)
    if n_kept == 0 or max_q < 2:
        logger.warning('  No correlation parameters to evaluate — skipping.')
        return [], {}

    # Selection: non-identity correlation matrix, and NUTS-converged where diagnostics exist.
    sel = correlatedMask(data_batch)
    n_corr = int(sel.sum())
    conv_mask = nutsConvergeMaskFromNpz(data_path, cap_mask, convergence_mode)
    if conv_mask is not None:
        sel &= conv_mask
        logger.info(
            '  Correlated: %d / %d; NUTS-converged (%s) among them: %d',
            n_corr,
            n_kept,
            convergence_mode,
            int(sel.sum()),
        )
    else:
        logger.info('  Correlated: %d / %d (no NUTS diagnostics)', n_corr, n_kept)
    n_sel = int(sel.sum())
    if n_sel == 0:
        logger.warning('  Empty selection — skipping.')
        return [], {}
    counts = {'total': n_total, 'kept': n_kept, 'correlated': n_corr, 'selected': n_sel}
    if conv_mask is not None:  # convergence rate on correlated vs. independent datasets (prose)
        corr = correlatedMask(data_batch)
        counts['conv_corr_pct'] = round(100 * conv_mask[corr].mean(), 1)
        counts['conv_uncorr_pct'] = round(100 * conv_mask[~corr].mean(), 1)

    # MB samples + refinements over the capacity-kept batch: same cache keys as the oracle run.
    proposal_mb, _ = loadOrSampleMB(
        model,
        data_batch,
        data_path,
        ckpt_dir,
        prefix,
        n_samples,
        batch_size,
        seed,
        device,
        cap_mask,
        warmup=warmup,
    )
    if rescale:
        proposal_mb.rescale(data_batch['sd_y'])
        data_batch = rescaleData(data_batch)
    refined: list[tuple[str, Proposal]] = []
    for method in validMethods(methods, lf):
        logger.info('  Refining MB with %s', method)
        p_ref, _ = loadOrRefine(
            method,
            proposal_mb,
            data_batch,
            data_path,
            ckpt_dir,
            prefix,
            n_samples,
            seed,
            lf,
            rescale,
            cap_mask,
            batch_size,
            device=device,
        )
        refined.append((f'MB+{method}', p_ref))

    sel_batch = subsetBatch(data_batch, sel)
    rows: list[dict] = []
    for label, proposal in [('MB', proposal_mb)] + refined:
        rows.append(buildRow(label, regime, corrMetrics(subsetProposal(proposal, sel), sel_batch)))
    del refined, proposal_mb
    gc.collect()

    # Reference methods, streamed one at a time (only one fit-tensor set resident).
    for label, method in (('NUTS', 'nuts'), ('ADVI', 'advi'), ('LA', 'laplace')):
        fit_batch, _, _, _ = loadRegimeBatch(
            data_path, max_d, max_q, exclude_prefixes=fitExcludePrefixes(method)
        )
        if f'{method}_corr_rfx' not in fit_batch:
            logger.info('  %s: no correlation fits in file — skipping.', label)
            del fit_batch
            gc.collect()
            continue
        success = fitBatchMask(fit_batch, method) & sel
        logger.info('  %s success within selection: %d / %d', label, int(success.sum()), n_sel)
        if not success.any():
            del fit_batch
            gc.collect()
            continue
        method_batch = subsetBatch(methodFitBatch(fit_batch, method), success)
        del fit_batch
        proposal = fit2proposal(method_batch, method)
        data_sub = subsetBatch(data_batch, success)
        if rescale:
            proposal.rescale(data_sub['sd_y'])
        rows.append(buildRow(label, regime, corrMetrics(proposal, data_sub)))
        del proposal, method_batch, data_sub
        gc.collect()

    return rows, counts


# ---------------------------------------------------------------------------
# Output


def saveTables(rows: list[dict], counts: dict[str, int], outdir: Path, run_name: str, dp: int):
    outdir.mkdir(parents=True, exist_ok=True)
    regime = rows[0]['regime']
    note = (
        f'correlation pairs only; datasets with a non-identity correlation matrix '
        f"(q>=2, eta>0) and NUTS-converged: {counts['selected']} of {counts['kept']} "
        f"(correlated: {counts['correlated']}); spread over correlation pairs"
    )
    if 'conv_corr_pct' in counts:
        note += (
            f"; NUTS converged on {counts['conv_corr_pct']}% of the correlated and "
            f"{counts['conv_uncorr_pct']}% of the independent datasets"
        )

    md_parts = [f'# Oracle Evaluation (correlation parameters): {run_name}', note]
    for stat in STATS:
        md_rows = [
            [regime, r['method']] + [_fmtMd(r['stats'][stat][c], dp) for c in METRICS] for r in rows
        ]
        table = tabulate(
            md_rows, headers=['regime', 'method'] + METRICS, tablefmt='pipe', stralign='right'
        )
        md_parts.append(f'## {stat}\n\n{table}')
    md_path = outdir / f'oracle_corr_{run_name}.md'
    md_path.write_text('\n\n'.join(md_parts) + '\n')
    logger.info('Saved Markdown → %s', md_path)

    header = r'$\mathrm{regime}$ & $\mathrm{model}$ & $r$ & $\mathrm{NRMSE}$ & $\mathrm{ECE}$ & $\mathrm{EACE}$ \\'
    for stat in STATS:
        lines = [
            f'% entries: {stat}; {note}',
            r'\begin{tabular}{cc|cccc}',
            r'    \toprule',
            f'    {header}',
            r'    \midrule',
        ]
        for j, row in enumerate(rows):
            regime_cell = rf'\texttt{{{regime}}}' if j == 0 else ''
            cells = ' & '.join(_fmtTex(row['stats'][stat][c], dp) for c in METRICS)
            lines.append(rf'      {regime_cell} & \texttt{{{row["method"]}}} & {cells} \\')
        lines += [r'    \bottomrule', r'\end{tabular}', '']
        suffix = '' if stat == PRIMARY else '_' + stat.split(' ')[0] + stat.split(' ')[-1]
        tex_path = outdir / f'oracle_corr_{run_name}{suffix}.tex'
        tex_path.write_text('\n'.join(lines))
        logger.info('Saved LaTeX → %s', tex_path)


def resolvePlotRows(rows: list[dict], wanted: list[str]) -> list[tuple[str, dict]]:
    """Map CLI labels to result rows: 'MB' → refined row if present (paper's MB), 'MB0' → raw."""
    by_label = {r['method']: r for r in rows}
    refined = [lbl for lbl in by_label if lbl.startswith('MB+')]
    out = []
    for w in wanted:
        if w == 'MB':
            key, title = (refined[0], 'MB') if refined else ('MB', 'MB')
        elif w == 'MB0':
            key, title = 'MB', r'MB$^0$'
        else:
            key, title = w, w
        if key in by_label:
            out.append((title, by_label[key]))
        else:
            logger.warning('  plot: no row for %s — skipped', w)
    return out


def plotCorrRecovery(rows: list[dict], q: int, outdir: Path, run_name: str) -> Path:
    """One recovery scatter per method, correlation pairs only."""
    n = len(rows)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 6), dpi=DPI, squeeze=False)
    names = getCorrRfxNames(q)
    for k, (ax, (title, row)) in enumerate(zip(axs[0], rows)):
        m = row['plot']
        _plotRecovery(
            ax,
            targets=m['targets'].numpy(),
            estimates=m['estimates'].numpy(),
            mask=m['pair_mask'].numpy(),
            stats={'r': row['r'][0], 'NRMSE': row['NRMSE'][0]},
            names=names,
            colors=paramColors(names),
            title=title,
            ylabel='Estimate' if k == 0 else '',
            upper=True,
            lower=True,
            show_legend=False,
        )
        ax.set_box_aspect(1)
    # up to q(q-1)/2 pairs: one shared legend strip below the panels instead of a box that
    # would cover the first panel's data
    handles, labels = axs[0, 0].get_legend_handles_labels()
    proxies = [legendProxy(h, lbl) for h, lbl in zip(handles, labels)]
    fig.legend(
        proxies,
        labels,
        loc='lower center',
        ncol=len(labels),
        fontsize=20,
        markerscale=1.5,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    path = savePlot(outdir, f'oracle_corr_{run_name}', ending='pdf')
    savePlot(outdir, f'oracle_corr_{run_name}', ending='png')
    plt.close(fig)
    logger.info('Saved figure → %s', path)
    return path


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)

    ckpt_dir = Path(cfg.checkpoint)
    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)
    max_d, max_q, lf = model_cfg.max_d, model_cfg.max_q, model_cfg.likelihood_family

    data_id = cfg.data_id
    regime = data_id.split('-')[0]
    stem = f'{ckpt_dir.name}_{data_id}'
    methods = cfg.methods if cfg.methods is not None else posthocDefaults(lf)

    data_path = DATA_DIR / data_id / 'test.fit.npz'
    base_path = DATA_DIR / data_id / 'test.npz'
    if not data_path.exists():
        logger.error('%s: test.fit.npz not found', data_id)
        return
    if not base_path.exists():
        base_path = data_path

    rows, counts = evaluateRegime(
        model,
        data_path,
        base_path,
        max_d,
        max_q,
        lf,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        device=device,
        regime=regime,
        ckpt_dir=ckpt_dir,
        prefix=cfg.prefix,
        seed=cfg.seed,
        methods=methods,
        rescale=cfg.rescale,
        convergence_mode=cfg.convergence_mode,
        warmup=cfg.warmup,
    )
    if not rows:
        logger.error('Nothing evaluated for %s.', data_id)
        return

    dp = cfg.decimals
    md_rows = [[regime, r['method']] + [_fmtMd(r[c], dp) for c in METRICS] for r in rows]
    print(f"\n{counts['selected']} correlated, NUTS-converged datasets of {counts['kept']}")
    print(tabulate(md_rows, headers=['regime', 'method'] + METRICS, tablefmt='simple'))

    outdir = Path(cfg.outdir)
    saveTables(rows, counts, outdir, f'{stem}_conv', dp)
    if cfg.plot:
        plot_rows = resolvePlotRows(rows, cfg.plot_methods)
        if plot_rows:
            plotCorrRecovery(plot_rows, max_q, outdir, f'{stem}_conv')


if __name__ == '__main__':
    main()
