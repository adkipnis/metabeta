"""Real-data comparison of the IMH under-coverage remedy (local moves) on the Normal-slope prior grid.

For every grid point of experiments/evaluation/prior_sensitivity.py (--rfx slope) that has a
reference NUTS fit, sample the posterior with Api.sample(refine=True) under each variant and
score it against that fit with PriorSensitivity.agreement (r, sd ratio, rank MAD, ΔLOO-NLL,
key-effect shift and 90% width ratio, flag) plus the 90% width ratio and |Δmedian| / NUTS sd of
every global parameter. The Api batches each (ffx family, SD family) block of priors in one
call, as the prior-grid experiment does. Nothing is written to the experiment's data dirs.

Variants are 'mb' (plain IMH, random_walk=False) or '+'-joined tokens: 'gate' (the '+moves' RW,
posterior_eval.MOVES_RW: local moves where the IMH min bulk ESS < 150), 'B<x>' (burn-in steps per
active global dimension) and 'K<n>' (IS² draws per group of the moves' target estimate).

    uv run python experiments/posthoc/imh_moves_real.py --variants mb gate gate+K2 --seeds 0 1
    uv run python experiments/posthoc/imh_moves_real.py --stage table
"""

import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from metabeta.utils.experiments import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'evaluation'))
from prior_sensitivity import DATASETS, PriorSensitivity  # noqa: E402

from metabeta.models.api import Api  # noqa: E402
from metabeta.utils.constants import LIKELIHOOD_FAMILIES  # noqa: E402
from metabeta.utils.dataloader import sliceBatch, toDevice  # noqa: E402
from metabeta.utils.posterior_eval import MOVES_RW  # noqa: E402
from metabeta.utils.sampling import setSeed  # noqa: E402

OUT = REPO_ROOT / 'metabeta' / 'outputs' / 'results' / 'imh_moves' / 'prior_grid_slope.csv'


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['run', 'table'], default='run')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--variants', nargs='+', default=['mb', 'gate'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--n-samples', type=int, default=4000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=None, help='Api refinement chunk size')
    p.add_argument('--max-points', type=int, default=None, help='cap per block (smoke tests)')
    p.add_argument('--out', default=str(OUT))
    return p.parse_args()
# fmt: on


def variantKwargs(name: str) -> dict:
    if name == 'mb':
        return {'random_walk': False}  # plain IMH
    moves = None
    gated = MOVES_RW  # the '+moves' RW; tokens override it
    for tok in name.split('+'):
        if tok == 'gate':
            moves = moves or gated
        elif tok.startswith('B'):
            moves = replace(moves or gated, burnin_per_dim=float(tok[1:]))
        elif tok.startswith('K'):
            moves = replace(moves or gated, n_inner=int(tok[1:]))
        else:
            raise ValueError(f'unknown variant token {tok!r} in {name!r}')
    return {'random_walk': moves}


def globalScores(ps: PriorSensitivity, name: str, p, nuts) -> dict:
    """90% width ratio and |Δmedian| / NUTS sd of every global parameter, summarized."""
    a, b = ps.draws(name, p), ps.draws(name, nuts)
    width = lambda x: np.quantile(x, 0.95) - np.quantile(x, 0.05)
    wr, sh = {}, {}
    for k in a:
        if k.startswith('u'):  # rfx
            continue
        wr[k] = width(a[k]) / max(width(b[k]), 1e-12)
        sh[k] = abs(np.median(a[k]) - np.median(b[k])) / max(b[k].std(), 1e-12)
    out = {f'wr_{k}': v for k, v in wr.items()} | {f'shift_{k}': v for k, v in sh.items()}
    w = np.array(list(wr.values()))
    return out | {
        'wr_min': float(w.min()),
        'wr_median': float(np.median(w)),
        'shift_max': float(max(sh.values())),
    }


def run(cfg: argparse.Namespace) -> None:
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out_path).to_dict('records') if out_path.exists() else []
    done = {(r['dataset'], r['variant'], r['seed']) for r in rows}
    ps = PriorSensitivity(argparse.Namespace(datasets=cfg.datasets, rfx='slope'))
    S = cfg.n_samples
    for name in cfg.datasets:
        meta, data, formula = ps.meta[name], ps.data[name], ps.formula(name)
        api = Api.from_pretrained(
            LIKELIHOOD_FAMILIES[meta['lf']], device=cfg.device, batch_size=cfg.batch_size
        )
        blocks = []
        for block in ps.blocks(name):
            pts = [
                i
                for i in block.index
                if (ps.dataDir(name) / 'fits' / f'test_nuts2_{i:03d}.npz').exists()
            ]
            blocks.append(block.loc[pts[: cfg.max_points]])
        blocks = [b for b in blocks if len(b)]
        if not blocks:
            print(f'{name}: no NUTS-referenced grid points, skipped', flush=True)
            continue
        refs: dict[int, tuple] = {}
        warm = {f'p{i:03d}': ps.prior(name, pt) for i, pt in blocks[0].iterrows()}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            api.sample(data, formula=formula, priors=warm, n_samples=S)  # untimed warm-up
        for seed in cfg.seeds:
            for variant in cfg.variants:
                if (name, variant, seed) in done:
                    continue
                kwargs = variantKwargs(variant)
                for block in blocks:
                    priors = {f'p{i:03d}': ps.prior(name, pt) for i, pt in block.iterrows()}
                    batch = toDevice(api.prepareData(data, formula=formula, priors=priors), 'cpu')
                    setSeed(seed)
                    t0 = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        res = api.sample(
                            data, formula=formula, priors=priors, n_samples=S, refine=True, **kwargs
                        )
                    wall = time.perf_counter() - t0
                    res.proposal.to('cpu')
                    sg = res.safeguards
                    for j, point in enumerate(block.index):
                        batch1 = sliceBatch(batch, j, j + 1)
                        if point not in refs:
                            fits = ps.loadFits(name, point, batch1)
                            refs[point] = fits['NUTS']
                        nuts, ndiag = refs[point]
                        p = res.proposal.slice_b(j, j + 1)
                        row = {
                            'dataset': name, 'point': int(point), 'variant': variant, 'seed': seed,
                            'tau_beta': float(block.loc[point, 'tau_beta']),
                            'nuts_converged': bool(ndiag['nuts_converged']),
                            'accept': float(sg['accept_rate'][j]), 'ess_min': float(sg['ess_min'][j]),
                            'moved': bool(sg['random_walk_ran'][j]),
                            'move_accept': float(sg['random_walk_accept'][j]),
                            'wall_s_per_point': wall / len(block),
                        }  # fmt: skip
                        row |= ps.agreement(name, p, nuts, batch1) | globalScores(ps, name, p, nuts)
                        rows.append(row)
                    print(
                        f'{name} {variant} seed {seed} block of {len(block)}: {wall:.1f} s',
                        flush=True,
                    )
                pd.DataFrame(rows).to_csv(out_path, index=False)
        del api
    table(cfg)


def table(cfg: argparse.Namespace) -> None:
    df = pd.read_csv(cfg.out)
    df = df[df['nuts_converged'].astype(bool)]
    agg = df.groupby(['dataset', 'variant']).agg(
        n=('point', 'size'),
        key_wr=('key_width_ratio', 'median'),
        key_wr_min=('key_width_ratio', 'min'),
        wr_min_q05=('wr_min', lambda s: s.quantile(0.05)),
        wr_lt_08=('wr_min', lambda s: (s < 0.8).mean()),
        shift_max_med=('shift_max', 'median'),
        flagged=('flagged', 'sum'),
        sigma_ratio=('sigma_ratio', 'median'),
        dloo=('delta_loo_nll', 'median'),
        moved=('moved', 'mean'),
        accept=('accept', 'mean'),
        s_per_point=('wall_s_per_point', 'mean'),
    )
    print(agg.round(3).to_markdown())


if __name__ == '__main__':
    cfg = setup()
    if cfg.stage == 'table':
        table(cfg)
    else:
        run(cfg)
