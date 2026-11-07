"""Group budget: metabeta against NUTS and the truth in every cell of
experiments/simulation/group_budget_sets.py (simulated, tags gb...) and against NUTS in every
cell of group_budget_real_sets.py (real data, tags gbr..., which have no truth, so
evaluateRegime and the oracle grid are skipped).

``run`` evaluates one (family, cell) with the benchmark functions, unchanged and with their
CLI defaults except 4000 draws and no latency cache: oracle_posterior.evaluateRegime scores
every method against the generating parameters, real_posterior.evaluateReal scores MB, its
refinement, ADVI and PF against the nuts2 reference, pooled and per parameter block. Their
tables are written as the two scripts write them, and both draw MB from one cache. Then
full_real_diagnostics.diagnose runs on every dataset: the safeguards the Api reports (IMH
acceptance, min ESS, pool k-hat against its thresholds), from a batch-of-one run as in the Api.

``collect`` lays the per-cell outputs out as grids per family and kind of data. Its only
statistic is real_posterior's median ± MAD over datasets, applied to the per-block values it
saved.

Usage from repo root:
    python experiments/evaluation/group_budget.py run --family n --cell gb200x3000
    python experiments/evaluation/group_budget.py run --family n --cell gbr200x3000
    python experiments/evaluation/group_budget.py collect
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tabulate import tabulate

from metabeta.models.api import IMH_ACCEPT_WARN, IMH_ESS_WARN, PSIS_K_THRESHOLD
from metabeta.utils.dataloader import Collection
from metabeta.utils.device import setDevice
from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, RESULTS_DIR
from metabeta.utils.logger import setupLogging
from metabeta.utils.posterior_eval import loadModel, posthocDefaults
from metabeta.utils.sampling import setSeed

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'simulation'))

import oracle_posterior  # noqa: E402
import real_posterior  # noqa: E402
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402
from full_real_diagnostics import diagnose  # noqa: E402
from group_budget_real_sets import SOURCES  # noqa: E402
from group_budget_real_sets import cellTag as realTag  # noqa: E402
from group_budget_sets import CELLS, SIZE, cellTag  # noqa: E402

OUT_DIR = RESULTS_DIR / 'group-budget'
FAMILIES = {'n': ('normal', 0), 'b': ('bernoulli', 1), 'p': ('poisson', 2)}
PREFIX = 'latest'
N_SAMPLES = 4000
BUDGET = (200, 3000)
BLOCKS = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
TAGS = {'simulated': cellTag, 'real': realTag}


def checkpoint(family: str) -> Path:
    name = FAMILIES[family][0]
    return _ckpt_dir(name, SIZE, BEST_SEEDS[(name, SIZE)])


def paths(family: str, tag: str) -> dict[str, Path]:
    ckpt = checkpoint(family).name
    data_id = f'{SIZE}-{family}-{tag}'
    return {
        'oracle_rows': OUT_DIR / f'oracle_{ckpt}_{data_id}_rows.json',
        'real': OUT_DIR / f'real_{ckpt}_{PREFIX}_{data_id}_per_dataset.npz',
        'diagnostics': OUT_DIR / f'diagnostics_{ckpt}_{data_id}.jsonl',
    }


# ---------------------------------------------------------------------------
# run


def run(cfg: argparse.Namespace) -> None:
    lf = FAMILIES[cfg.family][1]
    ckpt = checkpoint(cfg.family)
    data_id = f'{SIZE}-{cfg.family}-{cfg.cell}'
    data_path = DATA_DIR / data_id / 'test.npz'
    out = paths(cfg.family, cfg.cell)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = setDevice('cpu')
    model, mcfg = loadModel(ckpt, PREFIX, device)
    methods = posthocDefaults(lf)
    stem = f'{ckpt.name}_{data_id}'
    if not cfg.cell.startswith('gbr'):  # real data has no truth
        setSeed(cfg.seed)
        rows, rows_conv = oracle_posterior.evaluateRegime(
            model,
            data_path,
            mcfg.max_d,
            mcfg.max_q,
            lf,
            n_samples=N_SAMPLES,
            batch_size=cfg.batch_size,
            device=device,
            regime=SIZE,
            ckpt_dir=ckpt,
            prefix=PREFIX,
            seed=cfg.seed,
            methods=methods,
        )
        oracle_posterior.saveTables({SIZE: rows}, OUT_DIR, stem)
        if rows_conv:
            oracle_posterior.saveTables({SIZE: rows_conv}, OUT_DIR, f'{stem}_conv')
        out['oracle_rows'].write_text(json.dumps({'rows': rows, 'rows_conv': rows_conv}))

    setSeed(cfg.seed)
    real_rows = real_posterior.evaluateReal(
        model,
        data_path,
        mcfg.max_d,
        mcfg.max_q,
        lf,
        N_SAMPLES,
        cfg.batch_size,
        device,
        rescale=True,
        ckpt_dir=ckpt,
        prefix=PREFIX,
        seed=cfg.seed,
        methods=methods,
        latency=False,
    )
    real_posterior.savePerDataset(real_rows, OUT_DIR, f'{ckpt.name}_{PREFIX}', data_id)
    real_posterior.saveTables({SIZE: real_rows}, OUT_DIR, stem, PREFIX)

    setSeed(cfg.seed)
    col = Collection(data_path, max_d=mcfg.max_d, max_q=mcfg.max_q)
    with out['diagnostics'].open('w') as f:
        for i in range(len(col)):
            info, _, _ = diagnose(model, lf, col, i, N_SAMPLES)
            line = json.dumps({'data_id': data_id, 'index': i, **info})
            print(line, flush=True)
            f.write(line + '\n')


# ---------------------------------------------------------------------------
# collect


def fmt(values: np.ndarray, dp: int = 2) -> str:
    return real_posterior._fmtMd(real_posterior._medianMad(values), dp)


def cellLabel(m: int, n: int) -> list[str]:
    inside = m <= BUDGET[0] and n <= BUDGET[1]
    return [str(m), str(n), 'in' if inside else 'beyond']


def collectFamily(family: str, kind: str) -> str:
    head = ['groups', 'rows', 'budget', 'method']
    blocks = [b for b in BLOCKS if family == 'n' or b != 'sigma_eps']
    agree, sigma, rank, oracle, diag = [], [], [], [], []
    for m, n in CELLS:
        out = paths(family, TAGS[kind](m, n))
        label = cellLabel(m, n)

        npz = np.load(out['real'])
        for method in dict.fromkeys(k.split('/')[0] for k in npz.files):
            row = label + [method, str(len(npz[f'{method}/index']))]
            pooled = ('r', 'sigma_ratio', 'rank_mad', 'delta_nll')
            agree.append(row + [fmt(npz[f'{method}/{k}']) for k in pooled])
            sigma.append(row + [fmt(npz[f'{method}/{b}/sigma_ratio']) for b in blocks])
            rank.append(row + [fmt(npz[f'{method}/{b}/rank_mad']) for b in blocks])

        saved = json.loads(out['oracle_rows'].read_text()) if kind == 'simulated' else None
        rows = (saved['rows_conv'] or saved['rows']) if saved else []
        for r in rows:
            classes = [r['classes'].get(c, {}).get('NRMSE') for c in ('beta', 'sigma', 'alpha')]
            oracle.append(
                label
                + [r['method']]
                + ['NA' if v is None else f'{v:.2f}' for v in classes]
                + [
                    oracle_posterior._fmtMd(None if r[k] is None else tuple(r[k]), 2)
                    for k in ('NRMSE', 'EACE', 'LOO-NLL')
                ]
            )

        lines = [json.loads(s) for s in out['diagnostics'].read_text().splitlines()]
        keys = ('accept_rate', 'ess_min', 'psis_k_pool', 'accept_flag', 'ess_flag', 'psis_k_flag')
        info = {k: np.array([d[k] for d in lines], dtype=float) for k in keys}
        flags = np.stack([info[k] for k in ('accept_flag', 'ess_flag', 'psis_k_flag')]) > 0
        diag.append(
            label
            + [lines[0]['method'], str(len(lines))]
            + [fmt(info['accept_rate'], 3), fmt(info['ess_min'], 1), fmt(info['psis_k_pool'])]
            + [f'{100 * f.mean():.0f}%' for f in (*flags, flags.any(0))]
        )

    md = lambda rows, cols: tabulate(rows, headers=cols, tablefmt='pipe', stralign='right')
    name = FAMILIES[family][0]
    if kind == 'real':
        name = f'{name}, real data ({SOURCES[family].stem})'
    parts = [
        f'# Group budget, {name}: {SIZE} checkpoint {checkpoint(family).name}, s{N_SAMPLES}',
        '## Agreement with nuts2, pooled (real_posterior.py; median ± MAD over NUTS-converged '
        'datasets)\n\n' + md(agree, head + ['N', 'r', 'sigma ratio', 'rank-MAD', 'Delta LOO-NLL']),
        '## Sigma ratio per block\n\n' + md(sigma, head + ['N'] + blocks),
        '## Rank-MAD per block\n\n' + md(rank, head + ['N'] + blocks),
    ]
    if oracle:
        parts.append(
            '## Against the truth (oracle_posterior.py; NUTS-converged rows where they differ; '
            'NRMSE per class and pooled, EACE mean ± std over parameter dimensions, LOO-NLL '
            'median ± MAD)\n\n'
            + md(
                oracle,
                head + ['NRMSE beta', 'NRMSE sigma', 'NRMSE alpha', 'NRMSE', 'EACE', 'LOO-NLL'],
            )
        )
    parts += [
        f'## Safeguards (full_real_diagnostics.diagnose; flags: acceptance < {IMH_ACCEPT_WARN}, '
        f'min ESS < {IMH_ESS_WARN}, pool k-hat > {PSIS_K_THRESHOLD})\n\n'
        + md(
            diag,
            ['groups', 'rows', 'budget', 'method', 'N', 'acceptance', 'min ESS', 'pool k-hat']
            + ['accept flag', 'ESS flag', 'k-hat flag', 'any flag'],
        ),
    ]
    return '\n\n'.join(parts) + '\n'


def collect() -> None:
    for kind, tag in TAGS.items():
        for family in FAMILIES:
            if not all(paths(family, tag(m, n))['diagnostics'].exists() for m, n in CELLS):
                print(f'{family} ({kind}): cells missing - skipped')
                continue
            report = collectFamily(family, kind)
            suffix = '' if kind == 'simulated' else '_real'
            path = OUT_DIR / f'group_budget{suffix}_{family}.md'
            path.write_text(report)
            print(report)
            print(f'Saved -> {path}')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['run', 'collect'])
    parser.add_argument('--family', choices=list(FAMILIES))
    parser.add_argument('--cell', choices=[tag(m, n) for tag in TAGS.values() for m, n in CELLS])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    if cfg.stage == 'run':
        run(cfg)
    else:
        collect()


if __name__ == '__main__':
    main()
