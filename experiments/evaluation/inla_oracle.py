"""
E4: R-INLA next to MB, NUTS, ADVI and LA on the oracle benchmarks.

Runs oracle_posterior.evaluateRegime on every (family, size) set with the checkpoints behind
Api.from_pretrained (API_MODELS, prefix best, 4000 MB draws, seed 0), and writes the E4
deliverables to experiments/results/:

    inla_oracle_{n,b,p}_{size}.csv   one row per dataset and method: per-dataset RMSE by
                                     parameter class, LOO-NLL, wall time, failed/converged
                                     flags, m, n, q, m/q (and the INLA random-effect model)
    inla_oracle_{family}.tex         Table 1 layout on the NUTS-converged subset
    inla_oracle_byclass.tex          App. B.3 layout (beta/sigma/alpha NRMSE with r, LOO-NLL,
                                     time), averaged over the evaluated size regimes
    inla_oracle_lowmq.tex            Table 1 layout on the converged datasets in the lowest
                                     m/q quartile (fewest groups per random effect)
    inla_oracle_tables.md            all of the above in Markdown, plus INLA failure counts

INLA fits come from test.inla.npz (metabeta/simulation/inla.py, 4000 joint draws, one thread);
its wall times are per core on the machine that ran the fits, MB times on the evaluation
device, NUTS/ADVI/LA times from their fit files.

Usage (from repo root; GPU interactive node, matching the paper's MB timings):
    uv run python experiments/evaluation/inla_oracle.py --device cuda
    uv run python experiments/evaluation/inla_oracle.py --device cuda --sizes small --families n
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM
from oracle_posterior import METRICS, _fmtMd, _fmtTex, evaluateRegime
from metabeta.simulation.inla import INLA_DEFAULT_TIMEOUT_S
from metabeta.utils.device import setDevice
from metabeta.utils.experiments import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR
from metabeta.utils.logger import setupLogging
from metabeta.utils.posterior_eval import loadModel, posthocDefaults
from metabeta.utils.sampling import setSeed

logger = logging.getLogger(__name__)

# Sources of the released joint checkpoints (HF adkipnis/metabeta@v1, metabeta-{family}.pt,
# submodels[i]['source']), all with prefix best.
API_MODELS: dict[str, dict[str, str]] = {
    fam: {size: f'data={size}-{fam}-mixed_model=large_seed={seed}' for size, seed in seeds.items()}
    for fam, seeds in {
        'n': {'small': 13, 'medium': 14, 'large': 9, 'huge': 16},
        'b': {'small': 6, 'medium': 3, 'large': 4, 'huge': 8},
        'p': {'small': 4, 'medium': 11, 'large': 6, 'huge': 9},
    }.items()
}
FAMILY_NAMES = {'n': 'normal', 'b': 'bernoulli', 'p': 'poisson'}
FAMILY_LABELS = {'n': 'Gaussian', 'b': 'Bernoulli', 'p': 'Poisson'}
# Table methods in display order; 'MB' is the default pipeline (flow + family IMH), the raw
# flow ('MB0') only goes to the CSV, as in the paper.
TABLE_METHODS = ['MB', 'NUTS', 'ADVI', 'LA', 'INLA']
CSV_NAMES = {
    'MB0': 'mb0',
    'MB': 'mb',
    'NUTS': 'nuts',
    'ADVI': 'advi',
    'LA': 'laplace',
    'INLA': 'inla',
}


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='E4: INLA on the oracle benchmarks')
    parser.add_argument('--sizes',    nargs='+', default=['small', 'medium', 'large', 'huge'])
    parser.add_argument('--families', nargs='+', default=['n', 'b', 'p'], choices=list(LF_FROM_FAM))
    parser.add_argument('--device',   type=str, default='cpu')
    parser.add_argument('--prefix',   type=str, default='best')
    parser.add_argument('--n_samples',  type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed',     type=int, default=0)
    parser.add_argument('--outdir',   type=str, default=str(RESULTS_DIR))
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


# =============================================================================
# Pipeline
# =============================================================================


class InlaOracle:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.device = setDevice(cfg.device)
        self.outdir = Path(cfg.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        # rows[family][size][subset] -> list of oracle_posterior rows, relabelled
        self.rows: dict[str, dict[str, dict[str, list[dict]]]] = {}
        self.failures: list[dict] = []

    # -------------------------------------------------------------------------
    # evaluation

    def evaluate(self) -> None:
        for fam in self.cfg.families:
            self.rows[fam] = {}
            for size in self.cfg.sizes:
                self.rows[fam][size] = self._evaluateSet(fam, size)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

    def _evaluateSet(self, fam: str, size: str) -> dict[str, list[dict]]:
        cfg = self.cfg
        data_id = f'{size}-{fam}-sampled'
        data_dir = DATA_DIR / data_id
        ckpt_dir = CHECKPOINT_DIR / API_MODELS[fam][size]
        model, model_cfg = loadModel(ckpt_dir, cfg.prefix, self.device)
        lf = model_cfg.likelihood_family
        by_subset = evaluateRegime(
            model,
            data_dir / 'test.fit.npz',
            data_dir / 'test.npz',
            model_cfg.max_d,
            model_cfg.max_q,
            lf,
            n_samples=cfg.n_samples,
            batch_size=cfg.batch_size,
            device=self.device,
            regime=size,
            ckpt_dir=ckpt_dir,
            prefix=cfg.prefix,
            seed=cfg.seed,
            methods=posthocDefaults(lf),
        )
        del model
        default = f'MB+{posthocDefaults(lf)[0]}'
        relabel = {'MB': 'MB0', default: 'MB'}
        for rows in by_subset.values():
            for row in rows:
                row['method'] = relabel.get(row['method'], row['method'])
        self._writeCsv(fam, size, data_dir, by_subset)
        return by_subset

    # -------------------------------------------------------------------------
    # per-dataset CSV

    def _writeCsv(self, fam: str, size: str, data_dir: Path, by_subset: dict) -> None:
        with np.load(data_dir / 'test.npz') as raw:
            info = {k: raw[k] for k in ('m', 'n', 'q', 'eta_rfx')}
        conv_rows = {r['method']: r for r in by_subset.get('conv', [])}
        cap = next(r['mask'] for r in by_subset[''] if r['method'] == 'MB')
        # MB's conv-row mask is exactly the converged set; evaluateRegime drops the conv group
        # when every capacity-kept dataset converged
        conv = conv_rows['MB']['mask'] if 'MB' in conv_rows else cap
        inla_model = np.where((info['eta_rfx'] > 0) & (info['q'] == 2), 'iid2d', 'iid')

        frames = []
        for row in by_subset['']:
            idx = np.nonzero(row['mask'])[0]
            per = {k: v.numpy() for k, v in row['per_dataset'].items()}
            df = pd.DataFrame({'idx': idx, **per})
            full = pd.DataFrame({'idx': np.nonzero(cap)[0]}).merge(df, on='idx', how='left')
            full.insert(1, 'method', CSV_NAMES[row['method']])
            full['failed'] = ~np.isin(full['idx'], idx)
            frames.append(full)
            if row['method'] == 'INLA':
                self._countFailures(fam, size, data_dir, cap, full, inla_model)
        out = pd.concat(frames, ignore_index=True)
        out['m'] = info['m'][out['idx']]
        out['n'] = info['n'][out['idx']]
        out['q'] = info['q'][out['idx']]
        out['m_over_q'] = out['m'] / out['q']
        out['converged'] = conv[out['idx']]
        out['inla_re_model'] = np.where(out['method'] == 'inla', inla_model[out['idx']], '')
        path = self.outdir / f'inla_oracle_{fam}_{size}.csv'
        out.to_csv(path, index=False, float_format='%.6g')
        logger.info('Saved %s', path)

    def _countFailures(
        self, fam: str, size: str, data_dir: Path, cap: np.ndarray, full: pd.DataFrame, re_model
    ) -> None:
        with np.load(data_dir / 'test.inla.npz') as raw:
            wall = raw['inla_wall_s']
        failed = full['failed'].to_numpy()
        idx = full['idx'].to_numpy()
        self.failures.append(
            {
                'set': f'{size}-{fam}',
                'datasets': int(cap.sum()),
                'failed': int(failed.sum()),
                'timeout': int((failed & (wall[idx] >= INLA_DEFAULT_TIMEOUT_S - 1)).sum()),
                'iid2d (auto)': int((re_model[idx] == 'iid2d').sum()),
                'wall median [s]': float(np.median(wall[idx][~failed])),
                'wall max [s]': float(wall[idx][~failed].max()),
            }
        )

    # -------------------------------------------------------------------------
    # tables

    def _ordered(self, rows: list[dict]) -> list[dict]:
        by_method = {r['method']: r for r in rows}
        return [by_method[m] for m in TABLE_METHODS if m in by_method]

    def _metricTable(self, groups: dict[str, list[dict]], fmt, tex: bool) -> str:
        """Table 1 layout: one block of method rows per group (regime or family/regime)."""
        if not tex:
            body = [
                [g, r['method']] + [fmt(r[c]) for c in METRICS]
                for g, rows in groups.items()
                for r in rows
            ]
            return tabulate(body, headers=['regime', 'model'] + METRICS, tablefmt='pipe')
        lines = [
            r'\begin{tabular}{cc|cccccc}',
            r'    \toprule',
            r'    $\mathrm{regime}$ & $\mathrm{model}$ & $r$ & $\mathrm{NRMSE}$ & $\mathrm{ECE}$'
            r' & $\mathrm{EACE}$ & $\mathrm{LOO\text{-}NLL}$ & $\mathrm{time}$ \\',
        ]
        for g, rows in groups.items():
            lines.append(r'    \midrule')
            for j, r in enumerate(rows):
                lead = rf'\texttt{{{g}}}' if j == 0 else ''
                cells = ' & '.join(fmt(r[c]) for c in METRICS)
                lines.append(rf'      {lead} & \texttt{{{r["method"]}}} & {cells} \\')
        return '\n'.join(lines + [r'    \bottomrule', r'\end{tabular}', ''])

    def _byClass(self) -> tuple[str, str]:
        """App. B.3 layout: per-class NRMSE (r), LOO-NLL and time, averaged over regimes."""
        body_md, lines = [], [
            r'\begin{tabular}{ll|ccc|cc}',
            r'    \toprule',
            r'    $\mathrm{family}$ & $\mathrm{model}$ & $\boldsymbol\beta$: NRMSE ($r$)'
            r' & $\boldsymbol\sigma$: NRMSE ($r$) & $\boldsymbol\alpha$: NRMSE ($r$)'
            r' & $\mathrm{LOO\text{-}NLL}$ & $\mathrm{time\ [s]}$ \\',
        ]
        for fam, sizes in self.rows.items():
            per_method: dict[str, list[dict]] = {}
            for by_subset in sizes.values():
                for r in by_subset.get('conv', by_subset['']):
                    per_method.setdefault(r['method'], []).append(r)
            methods = [m for m in TABLE_METHODS if m in per_method]
            lines += [
                r'    \midrule',
                rf'      \multirow{{{len(methods)}}}{{*}}{{{FAMILY_LABELS[fam]}}}',
            ]
            for m in methods:
                rs = per_method[m]
                cls = [
                    tuple(
                        np.mean([float(np.nanmean(r['by_class'][c][k].numpy())) for r in rs])
                        for k in (0, 1)
                    )
                    for c in ('beta', 'sigma', 'alpha')
                ]
                loo = np.mean([r['LOO-NLL'][0] for r in rs])
                time = np.mean([r['time'][0] for r in rs])
                cells = [f'${e:.2f}$ (${c:.2f}$)' for e, c in cls] + [
                    f'${loo:.2f}$',
                    f'${time:.3g}$',
                ]
                lines.append(rf'        & \texttt{{{m}}} & ' + ' & '.join(cells) + r' \\')
                body_md.append(
                    [FAMILY_LABELS[fam], m]
                    + [f'{e:.2f} ({c:.2f})' for e, c in cls]
                    + [f'{loo:.2f}', f'{time:.3g}']
                )
        lines += [r'    \bottomrule', r'\end{tabular}', '']
        md = tabulate(
            body_md,
            headers=[
                'family',
                'model',
                'β NRMSE (r)',
                'σ NRMSE (r)',
                'α NRMSE (r)',
                'LOO-NLL',
                'time [s]',
            ],
            tablefmt='pipe',
        )
        return '\n'.join(lines), md

    def writeTables(self) -> None:
        md = ['# E4 tables: INLA on the oracle benchmarks', '']
        md += ['## INLA failures, timeouts and random-effect model', '']
        md += [tabulate(self.failures, headers='keys', tablefmt='pipe', floatfmt='.1f'), '']
        for fam, sizes in self.rows.items():
            for subset, title in (('conv', 'NUTS-converged'), ('', 'all datasets')):
                groups = {s: self._ordered(b.get(subset, b[''])) for s, b in sizes.items()}
                md += [
                    f'## {FAMILY_LABELS[fam]}, {title}',
                    '',
                    self._metricTable(groups, _fmtMd, False),
                    '',
                ]
                if subset == 'conv':
                    tex = self._metricTable(groups, _fmtTex, True)
                    (self.outdir / f'inla_oracle_{FAMILY_NAMES[fam]}.tex').write_text(tex)
        tex, md_class = self._byClass()
        (self.outdir / 'inla_oracle_byclass.tex').write_text(tex)
        md += ['## By parameter class (NUTS-converged, averaged over regimes)', '', md_class, '']
        groups = {
            f'{fam} {s}': self._ordered(b['lowmq'])
            for fam, sizes in self.rows.items()
            for s, b in sizes.items()
            if b.get('lowmq')
        }
        (self.outdir / 'inla_oracle_lowmq.tex').write_text(self._metricTable(groups, _fmtTex, True))
        md += [
            '## Lowest m/q quartile (NUTS-converged)',
            '',
            self._metricTable(groups, _fmtMd, False),
            '',
        ]
        path = self.outdir / 'inla_oracle_tables.md'
        path.write_text('\n'.join(md))
        logger.info('Saved tables to %s', self.outdir)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    oracle = InlaOracle(cfg)
    oracle.evaluate()
    oracle.writeTables()
