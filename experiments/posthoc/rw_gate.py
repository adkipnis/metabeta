"""Gate calibration of the random-walk moves (RW, posthoc/metropolis.py LocalMoves) after the IMH.

Every dataset is refined once by the default IMH (imhMarginal on Normal, the pseudo-marginal
imhPM on GLMMs; the Api layout: 4 × 1,025 pool draws, 4,000 kept), then by the RW on every
dataset (ungated) from those same IMH states, so the two refinements are paired. Recorded per
dataset are the statistics the IMH produces anyway, which a gate can use at no extra cost:

  ess        IMH bulk ESS of each global (one column per scored global; 'ess_min' over all
             active globals, correlations included: today's gate statistic)
  accept     mean IMH acceptance
  khat       PSIS k̂ of the pool's log weights, the IMH's own (Vehtari et al., 2024)
  n_active   number of active global dimensions

and both refinements are scored against the reference NUTS fit (nuts2): per global (ffx,
sigma_rfx, sigma_eps) the 90% interval width ratio and |Δmedian| / NUTS sd, plus (simulated
sets) whether the 90% interval holds the true value.

Target (stage 'analyze'). Width ratio r and shift δ map to the coverage of a 90% interval with
those errors under a Gaussian posterior, Φ(1.645 r − δ) − Φ(−1.645 r − δ), whose distance from
0.9 is the global's coverage error. A dataset needs the RW if the RW lowers some global's
coverage error by at least TAU. The RW targets the same posterior as the IMH, so the converse
event (the RW raises some global's error by TAU) can only come from noise, in the IMH, the RW or
the reference; its rate ('mirror') is the target's noise floor.

Rules use only the recorded IMH statistics. Their thresholds are fitted on the simulated (oracle)
test sets (the lowest fire rate whose recall reaches 95% or 90% in every family); today's gate
and the best fitted rules are then checked on the held-out real-data benchmark and the prior
grid: fire rate, false alarms (fires without need), misses (needs it does not fire on), and the
share of globals with width ratio < 0.8 and their mean coverage error when the RW runs only where
the rule fires. A sweep of the ess_min threshold shows the same per source. '--tau' sets TAU.

Sources: 'oracle' ({family}_{size}: the simulated test sets), 'real' ({size}-{f}-real: the
real-data benchmark), 'grid' (sleep, cbpp, salamanders: the slope prior grid of
experiments/evaluation/prior_sensitivity.py, its NUTS-referenced points). One CSV per source
goes to metabeta/outputs/results/rw_gate/ (resumable: finished sources are skipped).

    uv run python experiments/posthoc/rw_gate.py --sources oracle --families poisson --sizes small --n-datasets 16
    uv run python experiments/posthoc/rw_gate.py --device cuda --sources oracle real grid
    uv run python experiments/posthoc/rw_gate.py --stage analyze
"""

import argparse
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'posthoc'))
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'evaluation'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402
from imh_moves import (  # noqa: E402
    FAMILY_INITIAL,
    LIKELIHOOD_FAMILY,
    loadModel,
    loadReference,
    makeBatches,
    refDraws,
)
from prior_sensitivity import DATASETS, PriorSensitivity  # noqa: E402

from metabeta.models.api import IMH_BURNIN, IMH_ESS_WARN, IMH_N_CHAINS, Api  # noqa: E402
from metabeta.posthoc.metropolis import IS2_N_INNER, MetropolisSampler  # noqa: E402
from metabeta.utils.constants import LIKELIHOOD_FAMILIES  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped, sliceBatch, toDevice  # noqa: E402
from metabeta.utils.posterior_eval import MOVES_RW  # noqa: E402
from metabeta.utils.psis import psislw  # noqa: E402
from metabeta.utils.sampling import setSeed  # noqa: E402

OUT_DIR = REPO_ROOT / 'metabeta' / 'outputs' / 'results' / 'rw_gate'
FAMILIES = ('normal', 'bernoulli', 'poisson')
SIZES = ('small', 'medium', 'large', 'huge')
REAL_SETS = [f'{s}-{f}' for f in 'nbp' for s in SIZES if (s, f) != ('huge', 'p')]
TAU = 0.05  # coverage-error reduction (90% interval) that makes a dataset need the RW
Z90 = norm.ppf(0.95)


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['run', 'analyze'], default='run')
    p.add_argument('--sources', nargs='+', default=['oracle'], choices=['oracle', 'real', 'grid'])
    p.add_argument('--sizes', nargs='+', default=list(SIZES))
    p.add_argument('--families', nargs='+', default=list(FAMILIES))
    p.add_argument('--grid', nargs='+', default=list(DATASETS), help='prior-grid datasets')
    p.add_argument('--n-samples', type=int, default=4000, help='kept IMH draws per dataset')
    p.add_argument('--n-datasets', type=int, default=None, help='cap per set (default: all)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=str(OUT_DIR))
    p.add_argument('--tau', type=float, default=TAU, help='analyze: coverage-error reduction that counts as need')
    return p.parse_args()
# fmt: on


# ==============================================================================
# Paired refinement
# ==============================================================================


@torch.no_grad()
def refinePaired(pool, batch: dict, lf: int) -> dict:
    """IMH, then the RW on every dataset from the same IMH states.

    Mirrors MetropolisSampler.__call__ up to the rfx attachment, which the scores do not use.
    Returns the IMH and RW globals (b, S, D_g) on the CPU, the per-global IMH bulk ESS (b, D_g;
    NaN = inactive), the mean IMH acceptance (b,) and the pool's PSIS k̂ (b,).
    """
    sampler = MetropolisSampler(
        batch,
        n_chains=IMH_N_CHAINS,
        n_steps=pool.n_samples // IMH_N_CHAINS,
        burnin=IMH_BURNIN,
        mode='marginal' if lf == 0 else 'laplace',
        likelihood_family=lf,
        n_inner=0 if lf == 0 else IS2_N_INNER,
        moves=replace(MOVES_RW, gate_ess=None),  # the '+moves' RW, ungated
    )
    log_w = sampler._logWeights(pool)  # (b, s)
    sg, _, idx, accept = sampler._runChains(log_w, pool.samples_g, None)
    state = sampler._laplaceState(idx) if lf != 0 else {}
    sg_rw, _, diag = sampler._applyMoves(pool, sg, state)
    _, khat = psislw(log_w.double().cpu().numpy())
    return {
        'imh': sg.float().cpu(),
        'rw': sg_rw.float().cpu(),
        'ess': diag['ess'].numpy(),
        'accept': accept.mean(-1).cpu().numpy(),
        'move_accept': diag['move_accept'].numpy(),
        'khat': khat,
    }


def poolSize(n_samples: int) -> int:
    return IMH_N_CHAINS * (math.ceil(n_samples / IMH_N_CHAINS) + IMH_BURNIN)


# ==============================================================================
# Scores
# ==============================================================================


def scoreBatch(res: dict, names: list[list[str]], ref, truth=None, scale=None) -> list[dict]:
    """Rows per (dataset, scored global) of one batch.

    ``names[i]``: dataset i's scored globals in the D_g layout (ffx0.., sigma_rfx0..,
    sigma_eps; None = not scored); ``ref(i, name)``: NUTS draws (None: no reference) and
    ``ref.converged[i]``; ``truth[i][name]``: simulated value; ``scale[i]``: factor that maps
    the refinements' globals to the reference's scale.
    """
    rows = []
    for i, nm in enumerate(names):
        ess = res['ess'][i]
        base = {
            'ess_min': float(np.nanmin(ess)),
            'n_active': int(np.isfinite(ess).sum()),
            'accept': float(res['accept'][i]),
            'move_accept': float(res['move_accept'][i]),
            'khat': float(res['khat'][i]),
            'ref_converged': bool(ref.converged[i]),
        }
        for j, name in enumerate(nm):
            if name is None:
                continue
            r = ref(i, name)
            if r is None:
                continue
            r05, r50, r95 = np.quantile(r, [0.05, 0.5, 0.95])
            row = dict(base, local=i, param=name, ess=float(ess[j]))
            for method in ('imh', 'rw'):
                x = res[method][i, :, j].numpy() * (1.0 if scale is None else scale[i])
                a05, a50, a95 = np.quantile(x, [0.05, 0.5, 0.95])
                row[f'{method}_wr'] = (a95 - a05) / max(r95 - r05, 1e-12)
                row[f'{method}_shift'] = abs(a50 - r50) / max(r.std(), 1e-12)
                if truth is not None:
                    row[f'{method}_cov90'] = float(a05 <= truth[i][name] <= a95)
            rows.append(row)
    return rows


class Reference:
    """NUTS draws per (dataset, global) of one batch, and the convergence mask."""

    def __init__(self, draws: list[dict[str, np.ndarray]], converged: np.ndarray) -> None:
        self.draws, self.converged = draws, converged

    def __call__(self, i: int, name: str) -> np.ndarray | None:
        return self.draws[i].get(name)


def globalNames(d: int, q: int, lf: int, pool) -> list[str | None]:
    """A dataset's scored global names in the D_g layout of ``pool`` (None: padding)."""
    names = [f'ffx{j}' if j < d else None for j in range(pool.d)]
    names += [f'sigma_rfx{j}' if j < q else None for j in range(pool.q)]
    return names + (['sigma_eps'] if lf == 0 else [])


# ==============================================================================
# Sources
# ==============================================================================


class GateStudy:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, key: str, rows: list[dict], **cols) -> None:
        df = pd.DataFrame(rows).assign(**cols)
        df.to_csv(self.out_dir / f'{key}.csv', index=False)
        n = df['dataset'].nunique()
        print(
            f'[{key}] {n} datasets, RW accept median {df["move_accept"].median():.2f}', flush=True
        )

    def _done(self, key: str) -> bool:
        if (self.out_dir / f'{key}.csv').exists():
            print(f'[{key}] done', flush=True)
            return True
        return False

    # ------------------------------------------------------------------
    # Simulated test sets and the real-data benchmark (a test.npz with nuts2 fits)
    # ------------------------------------------------------------------

    def runSets(self, source: str) -> None:
        for family in self.cfg.families:
            for size in self.cfg.sizes:
                f = FAMILY_INITIAL[family]
                if source == 'oracle':
                    key, data_dir = f'oracle_{family}_{size}', f'{size}-{f}-sampled'
                elif f'{size}-{f}' in REAL_SETS:
                    key, data_dir = f'real_{family}_{size}', f'{size}-{f}-real'
                else:
                    continue
                if not self._done(key):
                    self._runSet(key, family, size, DATA_DIR / data_dir / 'test.npz', source)

    def _runSet(self, key: str, family: str, size: str, data_path: Path, source: str) -> None:
        cfg, lf = self.cfg, LIKELIHOOD_FAMILY[family]
        ckpt = _ckpt_dir(family, size, BEST_SEEDS[(family, size)]) / 'latest.pt'
        model = loadModel(ckpt, cfg.device)
        col = Collection(data_path, permute=False)
        n = len(col) if cfg.n_datasets is None else min(cfg.n_datasets, len(col))
        items = [col[i] for i in range(n)]
        ref = loadReference(data_path, n, lf)
        if ref is None:
            print(f'[{key}] no reference fit, skipped', flush=True)
            return
        setSeed(cfg.seed)
        rows = []
        for idx in makeBatches(items, cfg.batch_size, poolSize(cfg.n_samples)):
            batch = toDevice(dict(collateGrouped([items[k] for k in idx])), cfg.device)
            stats = model._dataStatistics(batch) if model.analytical_context else None
            pool = model.estimate(
                batch, n_samples=poolSize(cfg.n_samples), stats=stats, local=False
            )
            res = refinePaired(pool, batch, lf)
            names = [globalNames(int(items[k]['d']), int(items[k]['q']), lf, pool) for k in idx]
            draws = [{nm: refDraws(ref, k, nm) for nm in ns if nm} for k, ns in zip(idx, names)]
            truth = None
            if source == 'oracle':
                truth = [self._truth(items[k], lf) for k in idx]
            r = Reference(draws, ref['converged'][idx])
            for row in scoreBatch(res, names, r, truth):
                rows.append(row | {'dataset': idx[row.pop('local')]})
        self._write(key, rows, source=source, family=family, size=size)

    @staticmethod
    def _truth(item: dict, lf: int) -> dict[str, float]:
        d, q = int(item['d']), int(item['q'])
        out = {f'ffx{j}': float(item['ffx'][j]) for j in range(d)}
        out |= {f'sigma_rfx{j}': float(item['sigma_rfx'][j]) for j in range(q)}
        if lf == 0:
            out['sigma_eps'] = float(np.asarray(item['sigma_eps']).reshape(-1)[0])
        return out

    # ------------------------------------------------------------------
    # Prior grid (Api-prepared batches, one per (ffx family, SD family) block)
    # ------------------------------------------------------------------

    def runGrid(self) -> None:
        ps = PriorSensitivity(argparse.Namespace(datasets=self.cfg.grid, rfx='slope'))
        for name in self.cfg.grid:
            key = f'grid_{name}'
            if self._done(key):
                continue
            meta, data, formula = ps.meta[name], ps.data[name], ps.formula(name)
            lf = meta['lf']
            api = Api.from_pretrained(LIKELIHOOD_FAMILIES[lf], device=self.cfg.device)
            fits_dir = ps.dataDir(name) / 'fits'
            setSeed(self.cfg.seed)
            rows = []
            for block in ps.blocks(name):
                pts = [i for i in block.index if (fits_dir / f'test_nuts2_{i:03d}.npz').exists()]
                if not pts:
                    continue
                priors = {f'p{i:03d}': ps.prior(name, block.loc[i]) for i in pts}
                batch_cpu = toDevice(api.prepareData(data, formula=formula, priors=priors), 'cpu')
                routes, _ = api._routeBatch(batch_cpu)
                batch = toDevice(dict(batch_cpu), api.device)  # toDevice moves in place
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pool, _ = api._runRouted(
                        batch, routes, n_samples=poolSize(self.cfg.n_samples), local=False
                    )
                    res = refinePaired(pool, batch, lf)
                names = [globalNames(meta['d'], meta['q'], lf, pool)] * len(pts)
                draws, conv = [], []
                for j, point in enumerate(pts):
                    nuts, diag = ps.loadFits(name, point, sliceBatch(batch_cpu, j, j + 1))['NUTS']
                    draws.append(self._gridDraws(nuts, meta['d'], meta['q']))
                    conv.append(diag['nuts_converged'])
                scale = batch_cpu['sd_y'].reshape(-1).numpy() if lf == 0 else None
                for row in scoreBatch(res, names, Reference(draws, np.array(conv)), scale=scale):
                    j = row.pop('local')
                    rows.append(
                        row | {'dataset': int(pts[j]), 'tau_beta': block.loc[pts[j], 'tau_beta']}
                    )
            self._write(key, rows, source='grid', family=LIKELIHOOD_FAMILIES[lf], size=name)

    @staticmethod
    def _gridDraws(nuts, d: int, q: int) -> dict[str, np.ndarray]:
        out = {f'ffx{j}': nuts.ffx[0, :, j].numpy() for j in range(d)}
        out |= {f'sigma_rfx{j}': nuts.sigma_rfx[0, :, j].numpy() for j in range(q)}
        if nuts.has_sigma_eps:
            out['sigma_eps'] = nuts.sigma_eps[0].numpy()
        return out


# ==============================================================================
# Analysis
# ==============================================================================


def coverageError(wr: pd.Series, shift: pd.Series) -> np.ndarray:
    """|coverage − 0.9| of a 90% interval with width ratio wr and shift (NUTS sd) under a
    Gaussian posterior."""
    return np.abs(norm.cdf(Z90 * wr - shift) - norm.cdf(-Z90 * wr - shift) - 0.9)


def load(out_dir: Path, tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-global rows, per-dataset frame) of the NUTS-converged datasets."""
    df = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob('*.csv'))], ignore_index=True)
    df = df[df['ref_converged'].astype(bool)].copy()
    df['set'] = df['family'] + '_' + df['size']
    df['cls'] = df['param'].str.replace(r'\d+$', '', regex=True)
    df['gain'] = coverageError(df['imh_wr'], df['imh_shift']) - coverageError(
        df['rw_wr'], df['rw_shift']
    )
    df['need'], df['mirror'] = df['gain'] >= tau, df['gain'] <= -tau
    keys = ['source', 'set', 'dataset']
    ds = df.groupby(keys, sort=False).agg(
        family=('family', 'first'),
        size=('size', 'first'),
        ess_min=('ess_min', 'first'),
        n_active=('n_active', 'first'),
        accept=('accept', 'first'),
        khat=('khat', 'first'),
        need=('need', 'any'),
        mirror=('mirror', 'any'),
    )
    is_ffx = df['cls'] == 'ffx'
    ds['ess_ffx'] = df[is_ffx].groupby(keys, sort=False)['ess'].min().reindex(ds.index)
    ds['ess_var'] = df[~is_ffx].groupby(keys, sort=False)['ess'].min().reindex(ds.index)
    return df, ds.reset_index()


# gate statistics as scores that fire below a threshold
SCORES = {
    'ess_min': lambda d: d['ess_min'],
    'ess_min/D': lambda d: d['ess_min'] / d['n_active'],
    'ess_var': lambda d: d['ess_var'],
    'ess_ffx': lambda d: d['ess_ffx'],
    'accept': lambda d: d['accept'],
    '-khat': lambda d: -d['khat'],
}
PAIRS = [
    ('ess_min', 'accept'),
    ('ess_min', '-khat'),
    ('ess_var', 'ess_ffx'),
    ('ess_min/D', 'accept'),
]
RECALLS = (0.95, 0.9)
SWEEP = (0, 25, 50, 75, 100, 150, 200, np.inf)  # ess_min thresholds of the cost-accuracy sweep


class GateAnalysis:
    """Fits each rule's thresholds on the simulated sets and scores them on every set."""

    def __init__(self, out_dir: Path, tau: float) -> None:
        self.tau = tau
        self.df, self.ds = load(out_dir, tau)
        self.score = pd.DataFrame({k: f(self.ds) for k, f in SCORES.items()})
        self.oracle = (self.ds['source'] == 'oracle').to_numpy()
        self.lines: list[str] = []

    def go(self) -> None:
        self.targetTable()
        self.sweepTable()
        current = f'ess_min < {IMH_ESS_WARN}'
        rules = {current: {'ess_min': IMH_ESS_WARN}}
        fits = []
        for recall in RECALLS:
            best = None
            for keys in [(k,) for k in SCORES] + PAIRS:
                thr = self.fit(keys, recall)
                row = self.ruleRow(thr) | {
                    'recall target': recall,
                    'rule': ' & '.join(f'{k} < {t:.3g}' for k, t in thr.items()),
                }
                fits.append(row)
                if best is None or row['fire'] < best[0]['fire']:
                    best = (row, thr)
            rules[best[0]['rule']] = best[1]
        self.emit('## Rules fitted on the simulated sets (pooled; per-family recall >= target)')
        self.emit(pd.DataFrame(fits).set_index(['recall target', 'rule']).round(3).to_markdown())
        self.setTable(rules)

    def fire(self, thr: dict[str, float]) -> np.ndarray:
        fire = np.ones(len(self.ds), dtype=bool)
        for k, t in thr.items():
            fire &= (self.score[k] < t).to_numpy()
        return fire

    def fit(self, keys: tuple[str, ...], recall: float) -> dict[str, float]:
        """Thresholds with the lowest pooled oracle fire rate whose recall reaches ``recall`` in
        every family (thresholds from the oracle score quantiles; pairs on a 40×40 grid)."""
        o = self.ds[self.oracle]
        n_q = 400 if len(keys) == 1 else 40
        grids = [
            np.unique(np.quantile(self.score.loc[self.oracle, k].dropna(), np.linspace(0, 1, n_q)))
            for k in keys
        ]
        best, best_fire = None, np.inf
        for ts in np.stack(np.meshgrid(*grids), -1).reshape(-1, len(keys)):
            thr = dict(zip(keys, ts + 1e-9))  # fire on '<=' the quantile
            fire = self.fire(thr)[self.oracle]
            ok = all(
                fire[(o['family'] == f).to_numpy() & o['need'].to_numpy()].mean() >= recall
                for f in o['family'].unique()
            )
            if ok and fire.mean() < best_fire:
                best, best_fire = thr, fire.mean()
        return best

    def ruleRow(self, thr: dict[str, float], mask: np.ndarray | None = None) -> dict:
        mask = self.oracle if mask is None else mask
        fire, need = self.fire(thr)[mask], self.ds['need'].to_numpy()[mask]
        return {
            'fire': fire.mean(),
            'false alarm': (fire & ~need).mean(),
            'miss': (~fire & need).sum() / max(need.sum(), 1),
            'precision': (fire & need).sum() / max(fire.sum(), 1),
        }

    def gated(self, fire: np.ndarray) -> pd.DataFrame:
        """Per set, when the RW runs where ``fire``: the share of globals with width ratio < 0.8
        and their mean 90% coverage error."""
        key = pd.MultiIndex.from_frame(self.ds[['source', 'set', 'dataset']])
        rows = self.df.set_index(['source', 'set', 'dataset'])
        on = pd.Series(fire, index=key).reindex(rows.index).to_numpy()
        wr = rows['rw_wr'].where(on, rows['imh_wr'])
        shift = rows['rw_shift'].where(on, rows['imh_shift'])
        out = pd.DataFrame({'wr<0.8': wr < 0.8, 'cov. error': coverageError(wr, shift)})
        return out.groupby(level=['source', 'set'], sort=False).mean()

    def sweepTable(self) -> None:
        """Per source and ess_min threshold: fire rate, recall, and the accuracy of the globals
        (pooled over the source's sets; 0 = IMH only, inf = RW everywhere)."""
        need = self.ds['need'].to_numpy()
        rows = []
        for t in SWEEP:
            fire = self.fire({'ess_min': t})
            acc = self.gated(fire).groupby(level='source').mean()  # sets weigh equally
            for src in acc.index:
                m = (self.ds['source'] == src).to_numpy()
                rows.append({
                    'source': src, 'ess_min <': t, 'fire': fire[m].mean(),
                    'recall': fire[m & need].mean(), **acc.loc[src].to_dict(),
                })  # fmt: skip
        t = pd.DataFrame(rows).pivot(index='ess_min <', columns='source')
        self.emit('## Sweep of the ess_min threshold (sets pooled with equal weight per source)')
        self.emit(t.round(4).to_markdown())

    def targetTable(self) -> None:
        g = self.ds.assign(gated=self.ds['ess_min'] < IMH_ESS_WARN).groupby(
            ['source', 'set'], sort=False
        )
        t = pd.DataFrame({
            'n': g.size(),
            'need': g['need'].mean(),
            'mirror': g['mirror'].mean(),
            f'need | ess_min < {IMH_ESS_WARN}': g.apply(lambda d: d.loc[d['gated'], 'need'].mean()),
            f'need | ess_min >= {IMH_ESS_WARN}': g.apply(lambda d: d.loc[~d['gated'], 'need'].mean()),
        })  # fmt: skip
        self.emit(f"## Target: RW lowers some global's 90% coverage error by >= {self.tau}")
        self.emit(t.round(3).to_markdown())

    def setTable(self, rules: dict[str, dict[str, float]]) -> None:
        """Per set: fire / false alarm / miss of every rule, and the share of globals with width
        ratio < 0.8 under IMH only, RW everywhere, and each rule."""
        sets = self.ds[['source', 'set']].drop_duplicates()
        base = {
            'IMH only': np.zeros(len(self.ds), bool),
            'RW everywhere': np.ones(len(self.ds), bool),
        }
        acc = {k: self.gated(v) for k, v in base.items()}
        for name, thr in rules.items():
            rows = {}
            for _, (src, st) in sets.iterrows():
                mask = ((self.ds['source'] == src) & (self.ds['set'] == st)).to_numpy()
                rows[(src, st)] = self.ruleRow(thr, mask)
            acc[name] = self.gated(self.fire(thr))
            t = pd.DataFrame(rows).T.rename_axis(['source', 'set']).join(acc[name])
            self.emit(f'## {name}')
            self.emit(t.round(3).to_markdown())
        for metric in ('wr<0.8', 'cov. error'):
            self.emit(f'## {metric} of the globals (RW where the rule fires)')
            self.emit(pd.DataFrame({k: v[metric] for k, v in acc.items()}).round(4).to_markdown())

    def emit(self, text: str) -> None:
        print(text + '\n', flush=True)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.write_text('\n\n'.join(self.lines) + '\n')


if __name__ == '__main__':
    cfg = setup()
    if cfg.stage == 'analyze':
        analysis = GateAnalysis(Path(cfg.out_dir), cfg.tau)
        analysis.go()
        analysis.save(Path(cfg.out_dir) / f'analysis_tau{cfg.tau:g}.md')
        sys.exit()
    study = GateStudy(cfg)
    for source in cfg.sources:
        if source == 'grid':
            study.runGrid()
        else:
            study.runSets(source)
