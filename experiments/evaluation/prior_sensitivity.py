"""
experiments/evaluation/prior_sensitivity.py — E2: zero-shot prior sensitivity analysis on real data.

Three held-out real datasets (one per likelihood family: sleepstudy, cbpp, salamanders) are
analysed under a grid of analyst priors with the released checkpoints (eval only).  Each
(fixed-effect family, rfx-SD family) combination is one batched Api.sample call, because the
router takes one family combination per batch.  NUTS on a sub-grid is the reference.

Everything below was fixed on 2026-09-24, before any NUTS result.  Two changes followed the first
metabeta run (sleepstudy, cbpp): the draw count went from 1000 to 4000, because IMH acceptance fell
below 0.1 at tau_beta <= 0.076; and epil (the plan's Poisson set) was replaced by salamanders,
because every released Poisson submodel needs >= 5 observations per group and epil has 4.
salamanders enters with its key predictor only (mined), the rule the other two datasets follow:
with all 11 preprocessed predictors (d = 12, poisson-large submodel) IMH acceptance was 0.02-0.03
even at default-like priors in the pilot, and that model is not reported.
A third change, before any NUTS result: the grid is capped at the training hyper-prior range
(simulation/prior.py, hypersample; App. F.3), because E1 showed the network extrapolating and IMH
acceptance collapsing beyond it.  Points above the cap are dropped, not clipped, so the grid
covers exactly the training range and differs per dataset.
A fourth change, 2026-09-25, after the first full run (which mixed a correlated random slope for
sleepstudy with random intercepts elsewhere): every dataset gets the same rfx structure, run as two
variants (--rfx): 'slope' adds a correlated random slope to cbpp (period_4) and salamanders (wtemp;
mined is constant within sites, so a slope on it is not identified), 'intercept' drops sleepstudy's
random slope.  Each variant has its own data directory and result / figure files.

Model space.  Priors act on the preprocessed data that both the Api and NUTS see: sleepstudy y
and Days are standardised (sd_y = 1, so the Api's sd_y rescaling is the identity), cbpp keeps its
0/1 period dummies, salamanders its 0/1 mined dummy.  NUTS receives the
identical prior arrays (see `prior`), so no unit conversion sits between the two.

Grid, before the cap (288 points):
  tau_beta   12 log-spaced values 0.05 ... 5.0, one scale for every slope, location 0; the
             intercept keeps its Bambi default scale (2.5 / 1.5 / 2.5 for Normal / Bernoulli /
             Poisson), capped like the slopes (Poisson: 1.5): its natural size is the baseline
             rate, not an effect size, so shrinking it with the slopes would confound the
             key-effect curve with a baseline-rate conflict
  ffx family Normal, Student-t(5)
  tau_sigma  0.25, 1, 2.5, 5 (every rfx SD)
  SD family  half-Normal, half-Student-t(5), Exponential (mean tau_sigma)
  sigma_eps and LKJ eta at the Bambi defaults (utils/priors.bambiDefaultPriors)
Cap (Normal / Bernoulli / Poisson): tau_beta <= 4.0 / 3.0 / 1.5, tau_sigma <= 5.0 / 2.0 / 1.0,
leaving 264 / 120 / 108 points.
NUTS sub-grid = the first points of the grid: (i) all tau_beta x both ffx families x
{HN(tau_hn), Exp(1)}, tau_hn = 2.5 (the Bambi default) where the cap allows it, else the largest
capped tau_sigma (1.0); (ii) the two tightest slope scales (0.05, 0.076), where IMH acceptance
collapsed in the first metabeta run, at every SD prior: 84 / 56 / 52 points.  Rule for the rest,
stated before the run: every other point with MB acceptance below the Api's warning threshold
(IMH_ACCEPT_WARN) is written to nuts_extra.txt for a second NUTS array; those points are selected
by the outcome, so they are reported separately from the sub-grid.

NUTS reliability = the paper's filter, utils/evaluation.nutsConvergeMask in 'liberal' mode as in
oracle_posterior.py (strict counts reported alongside).  Primary medians and flag counts are over
converged sub-grid points; all-point counts are reported too; unconverged NUTS points are drawn
hollow grey.  Built-in diagnostics per point: IMH acceptance and the PSIS k-hat of the flow pool
under the Api's IS correction; a point 'fires' at acceptance < 0.2 or k-hat > 0.7.

Disagreement flag (key effect, per NUTS point): |median_MB - median_NUTS| / sd_NUTS > 0.25, or
90% interval width ratio MB / NUTS outside [0.8, 1.25].

Figures (slice choice fixed before the run): row 1 = key effect vs tau_beta at SD prior
HN(tau_hn) in the main figure (prior_sensitivity.pdf) and Exp(1) in the appendix backup
(prior_sensitivity_exp1.pdf); row 2 (both figures) = random-intercept SD vs tau_sigma at
Normal(0, tau_beta), tau_beta the grid value nearest the Bambi default 2.5 (2.16 / 2.16 / 1.42).
Lines = MB (flow + IMH, the paper default) with markers shaded by IMH acceptance, dots = NUTS;
x-ranges differ because the grid does.

Stages (--stages, run in the order given):
  export    the grid as a fit.py batch: outputs/data/e2-{ds}-{rfx}/test.npz + config.yaml (no model)
  mb        MB0 (raw flow) and MB (flow + IMH) on all grid points, cached with per-batch
            wall-clock to outputs/data/e2-{ds}-{rfx}/mb_{device}.pt (loads the released checkpoint)
  analyze   MB cache + NUTS (and ADVI, if present) fits -> results/prior_sensitivity_{rfx}_{ds}.csv/.md
            and the appendix table prior_sensitivity_{rfx}_agreement.md/.tex
  evidence  Normal datasets only: IS log p(D) from the flow pool on all grid points vs Meng-Wong
            bridge sampling on the NUTS draws (experiments/posthoc/evidence.py)
  plot      prior_sensitivity_{rfx}.pdf (+ _main, _exp1; _bf after evidence)

Usage (from repo root):
    uv run python experiments/evaluation/prior_sensitivity.py --rfx slope --stages export mb analyze --dry_run
    uv run python experiments/evaluation/prior_sensitivity.py --rfx slope --stages export
    n=$(grep -c ',True$' metabeta/outputs/data/e2-sleep-slope/grid.csv)    # cluster: NUTS sub-grid size
    sbatch --array=0-$((n - 1)) scripts/fit-nuts-prior-grid.sh --dataset sleep --rfx slope
    uv run python experiments/evaluation/prior_sensitivity.py --rfx slope --stages mb --device cuda
    uv run python experiments/evaluation/prior_sensitivity.py --rfx slope --device cuda \
        --stages analyze evidence plot
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate

from metabeta.evaluation.predictive import getPosteriorPredictive, psisLooNLL
from metabeta.models.api import IMH_ACCEPT_WARN, PSIS_K_THRESHOLD
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.utils.api import coercePriors, parseFormula, resolveLikelihoodFamily
from metabeta.utils.constants import FFX_FAMILIES, LIKELIHOOD_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.dataloader import sliceBatch, toDevice
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.experiments import DATA_DIR, PREPROCESSED_DATA_DIR, REPO_ROOT, RESULTS_DIR
from metabeta.utils.logger import setupLogging
from metabeta.utils.plot import PALETTE, VECTOR_RASTER_DPI
from metabeta.utils.priors import bambiDefaultPriors
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed

# sibling experiment scripts (this directory is sys.path[0] at run time)
from real_posterior import computeCorr, computeRankMAD, computeSigmaRatio

logger = logging.getLogger(__name__)

# ==============================================================================
# Globals
# ==============================================================================

# fixed effects = the key predictor(s) of each dataset (sleepstudy: Days, cbpp: the period factor,
# salamanders: mining); rfx structure per --rfx: 'slope' = random intercept + one correlated random
# slope everywhere (sleepstudy's structure in the plan), 'intercept' = random intercept only
DATASETS = {
    'sleep': {
        'file': 'sleep__grp_group',
        'formula': {
            'slope': 'y ~ days + (1 + days | group)',
            'intercept': 'y ~ days + (1 | group)',
        },
        'key': 'days',
        'title': 'sleepstudy (Normal)',
        'key_label': r'$\beta_{\mathrm{Days}}$',
    },
    'cbpp': {
        'file': 'cbpp__grp_group',
        'formula': {
            'slope': 'y ~ period_4 + period_2 + period_3 + (1 + period_4 | group)',
            'intercept': 'y ~ period_2 + period_3 + period_4 + (1 | group)',
        },
        'key': 'period_4',
        'title': 'cbpp (Bernoulli)',
        'key_label': r'$\beta_{\mathrm{period\,4}}$',
    },
    # replaces epil, which no released Poisson submodel routes (4 observations per group < 5)
    'salamanders': {
        'file': 'salamanders__grp_group',
        # mined is a site-level covariate (constant within every site), so a random slope on it
        # is not identified; the slope variant puts it on water temperature, which varies within
        # every site, and keeps mined as the key effect
        'formula': {
            'slope': 'y ~ wtemp + mined_yes + (1 + wtemp | group)',
            'intercept': 'y ~ mined_yes + (1 | group)',
        },
        'key': 'mined_yes',
        'title': 'salamanders (Poisson)',
        'key_label': r'$\beta_{\mathrm{mined}}$',
    },
}

TAU_BETA = np.geomspace(0.05, 5.0, 12)
TAU_SIGMA = (0.25, 1.0, 2.5, 5.0)
NUTS_HN_TAU = 2.5  # Bambi default; replaced by the largest capped tau_sigma where above the cap
NUTS_EXP_TAU = 1.0
TIGHT_TAU_BETA = TAU_BETA[:2]  # 0.05, 0.076: IMH acceptance collapsed here in the first run
# largest tau_ffx / tau_rfx the training hyper-prior draws (simulation/prior.py, hypersample);
# the grid is capped at these
TRAIN_MAX_TAU_FFX = {0: 4.0, 1: 3.0, 2: 1.5}
TRAIN_MAX_TAU_RFX = {0: 5.0, 1: 2.0, 2: 1.0}

METHODS = {'MB0': False, 'MB': True}  # label -> Api.sample(refine=...)
FIT_PREFIXES = {'NUTS': 'nuts', 'ADVI': 'advi'}
QUANTILES = {'q05': 0.05, 'q50': 0.5, 'q95': 0.95}
NUTS_CHAINS = 4  # fit.py default; chains run in parallel, one core each

KEY_Z_MAX = 0.25
CONVERGENCE_MODE = 'liberal'  # nutsConvergeMask mode, as in oracle_posterior.py
NEAR_ZERO = 0.2  # |key effect| below this counts as the prior's spike at zero
FIRE_ACCEPT, FIRE_KHAT = 0.2, PSIS_K_THRESHOLD  # a built-in diagnostic fires beyond these
WIDTH_RATIO_RANGE = (0.8, 1.25)
FIG_SD_FAMILY = {  # figure suffix -> row-1 SD family, at that family's NUTS sub-grid scale
    '': 'halfnormal',
    '_exp1': 'exponential',
}
BAMBI_TAU = 2.5  # row 2 and the Bayes-factor reference use the grid tau_beta nearest to this

SD_MARKERS = {'halfnormal': 'o', 'exponential': 'D'}  # NUTS SD priors in figure row 2
FIG_DIR = Path.home() / 'LaTeX' / 'metabeta-iclr' / 'figures'
C_MB, C_NUTS = PALETTE[4], PALETTE[3]  # runtimes figure colours of MB and NUTS
C_UNCONV = '0.6'  # NUTS points failing nutsConvergeMask
# IMH acceptance of the MB markers: sequential in MB's hue, light end cut so low values stay visible
ACCEPT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'accept', plt.get_cmap('Purples')(np.linspace(0.3, 1.0, 256))
)
ACCEPT_NORM = mcolors.Normalize(0.0, 1.0)


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', nargs='+', required=True, choices=['export', 'mb', 'analyze', 'evidence', 'plot'])
    parser.add_argument('--rfx', type=str, required=True, choices=['slope', 'intercept'], help='rfx structure of every dataset: correlated random slope or random intercept only')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument('--device', type=str, default='cpu', help='device of the mb stage; analyze reads that cache')
    parser.add_argument('--n_samples', type=int, default=4000, help='posterior draws per prior (MB0 and MB)')
    parser.add_argument('--batch_size', type=int, default=None, help='Api refinement chunk size (bounds IMH memory)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_bridge', type=int, default=4000, help='bridge-sampling proposal draws')
    parser.add_argument('--fig_dir', type=str, default=str(FIG_DIR))
    parser.add_argument('--dry_run', action='store_true', help='build grid and data, log the plan, write nothing')
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


def nutsSdPriors(lf: int) -> dict[str, float]:
    """SD family -> tau_sigma of the NUTS sub-grid, inside the cap of likelihood family lf."""
    tau_hn = max(ts for ts in TAU_SIGMA if ts <= min(NUTS_HN_TAU, TRAIN_MAX_TAU_RFX[lf]))
    return {'halfnormal': tau_hn, 'exponential': NUTS_EXP_TAU}


def buildGrid(lf: int) -> pd.DataFrame:
    """The capped priors of likelihood family lf; the NUTS points come first, so a SLURM
    array 0..n_nuts-1 covers them."""
    rows = [
        {'ffx_family': f, 'sigma_family': s, 'tau_sigma': ts, 'tau_beta': tb}
        for f in FFX_FAMILIES
        for s in SIGMA_FAMILIES
        for ts in TAU_SIGMA
        for tb in TAU_BETA
        if tb <= TRAIN_MAX_TAU_FFX[lf] and ts <= TRAIN_MAX_TAU_RFX[lf]
    ]
    grid = pd.DataFrame(rows)
    nuts = nutsSdPriors(lf)
    grid['nuts_subgrid'] = [
        nuts.get(s) == ts or np.isclose(tb, TIGHT_TAU_BETA).any()
        for s, ts, tb in zip(grid.sigma_family, grid.tau_sigma, grid.tau_beta)
    ]
    grid = grid.sort_values('nuts_subgrid', ascending=False, kind='stable').reset_index(drop=True)
    grid.index.name = 'point'
    return grid


def nutsConverged(fit: dict[str, np.ndarray], mode: str) -> bool:
    """nutsConvergeMask on one per-point fit file."""
    keys = ('nuts_divergences', 'nuts_rhat', 'nuts_ess', 'nuts_ess_tail', 'nuts_max_treedepth')
    diag = {k: torch.as_tensor(fit[k])[None] for k in keys}
    diag['nuts_draws'] = torch.as_tensor(int(fit['nuts_draws']))
    return bool(nutsConvergeMask(diag, mode=mode)[0])


def flagCount(g: pd.DataFrame) -> str:
    """Flagged / compared points (points without a NUTS fit are not compared)."""
    return f'{int(g.flagged.fillna(False).astype(bool).sum())}/{int(g.flagged.notna().sum())}'


def medMad(x: pd.Series, dp: int = 2) -> str:
    a = x.dropna().to_numpy(dtype=float)
    if len(a) == 0:
        return 'NA'
    med = np.median(a)
    return f'{med:.{dp}f} ± {np.median(np.abs(a - med)):.{dp}f}'


# ==============================================================================
# Pipeline
# ==============================================================================


class PriorSensitivity:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.data: dict[str, dict[str, np.ndarray]] = {}
        self.meta: dict[str, dict] = {}
        for name in cfg.datasets:
            self._load(name)

    def _load(self, name: str) -> None:
        spec = DATASETS[name]
        with np.load(
            PREPROCESSED_DATA_DIR / 'test' / f'{spec["file"]}.npz', allow_pickle=True
        ) as f:
            data = dict(f)
        columns = tuple(str(c) for c in data['columns'])
        formula = parseFormula(self.formula(name))
        lower = [c.lower() for c in columns]  # the formula parser lowercases terms
        x_idx = [lower.index(t) for t in formula.fixed_terms]  # ValueError: term not a column
        if 'sd_y' in data:
            raise ValueError(f'{name}: the grid is defined for sd_y = 1 (no sd_y key)')
        rfx_names = ['Intercept' if t == '1' else t for t in formula.random_terms]
        if list(formula.random_terms) != ['1', *formula.fixed_terms[: len(rfx_names) - 1]]:
            # fit.py takes the rfx design as the first q columns of X (Z = X[:, :q])
            raise ValueError(f'{name}: random slopes must be the leading fixed terms')
        lf = resolveLikelihoodFamily(None, str(data['y_type']))
        grid = buildGrid(lf)
        self.data[name] = data
        self.meta[name] = {
            'grid': grid,
            'nuts_sd': nutsSdPriors(lf),
            'fig_tau_beta': min(grid.tau_beta.unique(), key=lambda tb: abs(tb - BAMBI_TAU)),
            'lf': lf,
            'd': len(x_idx) + 1,
            'x_idx': x_idx,
            'q': len(rfx_names),
            'm': int(data['m']),
            'n': int(data['n']),
            'ffx_names': ['Intercept', *(columns[i] for i in x_idx)],
            'rfx_names': rfx_names,
            'tau_intercept': min(
                float(bambiDefaultPriors(1, 1, lf)['tau_ffx'][0]), TRAIN_MAX_TAU_FFX[lf]
            ),
        }

    def formula(self, name: str) -> str:
        return DATASETS[name]['formula'][self.cfg.rfx]

    def dataDir(self, name: str) -> Path:
        return DATA_DIR / f'e2-{name}-{self.cfg.rfx}'

    def resultPath(self, suffix: str) -> Path:
        """Result files and figures of this rfx variant: prior_sensitivity_{rfx}{suffix}."""
        return RESULTS_DIR / f'prior_sensitivity_{self.cfg.rfx}{suffix}'

    def prior(self, name: str, point: pd.Series) -> dict[str, np.ndarray]:
        """Canonical prior arrays of one grid point: exactly what the Api and NUTS receive."""
        meta = self.meta[name]
        d, q = meta['d'], meta['q']
        values = coercePriors(
            {
                'nu_ffx': np.zeros(d),
                'tau_ffx': np.r_[meta['tau_intercept'], np.full(d - 1, point.tau_beta)],
                'family_ffx': FFX_FAMILIES.index(point.ffx_family),
                'tau_rfx': np.full(q, point.tau_sigma),
                'family_sigma_rfx': SIGMA_FAMILIES.index(point.sigma_family),
            },
            d=d,
            q=q,
            likelihood_family=meta['lf'],
        )
        values.pop('likelihood_family')
        return values

    def blocks(self, name: str) -> list[pd.DataFrame]:
        """One block per (ffx family, SD family): one batched Api call each."""
        grid = self.meta[name]['grid']
        return [b for _, b in grid.groupby(['ffx_family', 'sigma_family'], sort=False)]

    def logPlan(self) -> None:
        for name, meta in self.meta.items():
            grid = meta['grid']
            logger.info(
                '%s: %s | lf=%d d=%d q=%d m=%d n=%d | %d grid points, %d NUTS (%s), blocks %s | '
                'tau_beta %s, tau_sigma %s, intercept tau %g, figure tau_beta %.3g',
                name, self.formula(name), meta['lf'], meta['d'], meta['q'], meta['m'],
                meta['n'], len(grid), grid.nuts_subgrid.sum(), meta['nuts_sd'],
                [len(b) for b in self.blocks(name)], np.round(grid.tau_beta.unique(), 3).tolist(),
                grid.tau_sigma.unique().tolist(), meta['tau_intercept'], meta['fig_tau_beta'],
            )  # fmt: skip

    # --------------------------------------------------------------------------
    # export

    def export(self) -> None:
        """Write the grid as a fit.py batch (identical data, one prior per row)."""
        for name, data in self.data.items():
            meta = self.meta[name]
            n, d, q, m = meta['n'], meta['d'], meta['q'], meta['m']
            base = {
                'X': np.concatenate(
                    [np.ones((n, 1)), data['X'][:, meta['x_idx']]], axis=1
                ),  # (n, d)
                'y': data['y'].astype(np.float64),
                'groups': data['groups'],
                'ns': data['ns'],
                'n': np.array(n),
                'm': np.array(m),
                'd': np.array(d),
                'q': np.array(q),
                'sd_y': np.array(1.0),
                'ffx': np.full(d, np.nan),
                'sigma_rfx': np.full(q, np.nan),
                'rfx': np.full((m, q), np.nan),
                'corr_rfx': np.full((q, q), np.nan),
                'likelihood_family': np.array(meta['lf']),
            }
            priors = [self.prior(name, point) for _, point in meta['grid'].iterrows()]
            batch = {k: np.stack([v] * len(priors)) for k, v in base.items()}
            batch |= {k: np.stack([p[k] for p in priors]) for k in priors[0]}
            out = self.dataDir(name)
            logger.info('export %s: %s', out / 'test.npz', {k: v.shape for k, v in batch.items()})
            if self.cfg.dry_run:
                continue
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / 'test.npz', **batch)
            config = {'data_id': out.name, 'likelihood_family': meta['lf']}
            (out / 'config.yaml').write_text(yaml.safe_dump(config))
            meta['grid'].to_csv(out / 'grid.csv')

    # --------------------------------------------------------------------------
    # mb

    def sampleMB(self) -> None:
        from metabeta.models.api import Api

        S = self.cfg.n_samples
        for name, data in self.data.items():
            meta = self.meta[name]
            formula = self.formula(name)
            api_priors = [
                {f'p{i:03d}': self.prior(name, point) for i, point in block.iterrows()}
                for block in self.blocks(name)
            ]
            if self.cfg.dry_run:
                logger.info(
                    'mb %s: calls of %s priors x %d draws x %s',
                    name,
                    [len(p) for p in api_priors],
                    S,
                    list(METHODS),
                )
                continue
            api = Api.from_pretrained(
                LIKELIHOOD_FAMILIES[meta['lf']],
                device=self.cfg.device,
                batch_size=self.cfg.batch_size,
            )
            # untimed warm-up on this design (flow + IMH), so the first timed block pays no
            # one-off kernel or allocation cost
            api.sample(data, formula=formula, priors=api_priors[0], n_samples=S)
            blocks = []
            for block, priors in zip(self.blocks(name), api_priors):
                batch = toDevice(api.prepareData(data, formula=formula, priors=priors), 'cpu')
                self._checkPriors(batch, block, meta)
                entry = {'points': block.index.to_numpy(), 'batch': batch}
                for method, refine in METHODS.items():
                    setSeed(self.cfg.seed)
                    t0 = time.perf_counter()
                    res = api.sample(
                        data, formula=formula, priors=priors, n_samples=S, refine=refine
                    )
                    res.proposal.to('cpu')
                    wall = time.perf_counter() - t0
                    p = res.proposal
                    entry[method] = {
                        'wall_s': wall,
                        'data': {src: dict(inner) for src, inner in p.data.items()},
                        'd_corr': p.d_corr,
                        'has_sigma_eps': p.has_sigma_eps,
                        'accept_rate': (res.safeguards or {}).get('accept_rate'),
                        'map_z': (res.safeguards or {}).get('map_z'),
                        'route': res.routes[0],
                    }
                    logger.info(
                        '%s %s/%s %s: %.2f s',
                        name,
                        *block.iloc[0][['ffx_family', 'sigma_family']],
                        method,
                        wall,
                    )
                blocks.append(entry)
            self._writeNutsExtra(name, blocks)
            out = self.dataDir(name) / f'mb_{self.cfg.device}.pt'
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'device': self.cfg.device,
                    'n_samples': S,
                    'batch_size': self.cfg.batch_size,
                    'grid': meta['grid'],
                    'blocks': blocks,
                },
                out,
            )
            logger.info('saved %s', out)

    def _writeNutsExtra(self, name: str, blocks: list[dict]) -> None:
        """Points outside the NUTS sub-grid with MB acceptance below the Api's warning threshold,
        as an sbatch --array list (empty file: none)."""
        grid = self.meta[name]['grid']
        points = np.concatenate([b['points'] for b in blocks])
        accept = np.concatenate([b['MB']['accept_rate'] for b in blocks])
        extra = sorted(int(i) for i in points[accept < IMH_ACCEPT_WARN] if not grid.nuts_subgrid[i])
        (self.dataDir(name) / 'nuts_extra.txt').write_text(','.join(map(str, extra)))
        logger.info(
            '%s: %d low-acceptance points outside the NUTS sub-grid: %s', name, len(extra), extra
        )

    def _checkPriors(self, batch: dict, block: pd.DataFrame, meta: dict) -> None:
        """The Api batch rows must carry the block's priors in order (NUTS gets the same)."""
        d, q = meta['d'], meta['q']
        tau_ffx = np.repeat(block.tau_beta.to_numpy()[:, None], d, 1)
        tau_ffx[:, 0] = meta['tau_intercept']
        expect = {
            'tau_ffx': tau_ffx,
            'tau_rfx': np.repeat(block.tau_sigma.to_numpy()[:, None], q, 1),
            'family_ffx': block.ffx_family.map(FFX_FAMILIES.index).to_numpy(),
            'family_sigma_rfx': block.sigma_family.map(SIGMA_FAMILIES.index).to_numpy(),
        }
        for key, want in expect.items():
            got = batch[key].numpy()
            got = got[:, :d] if key == 'tau_ffx' else got[:, :q] if key == 'tau_rfx' else got
            if not np.allclose(got.reshape(want.shape), want):
                raise ValueError(f'Api batch {key} does not match the grid block')

    # --------------------------------------------------------------------------
    # analyze

    def analyze(self) -> None:
        frames = {}
        for name in self.data:
            cache_path = self.dataDir(name) / f'mb_{self.cfg.device}.pt'
            if self.cfg.dry_run:
                fits = sorted((self.dataDir(name) / 'fits').glob('test_*_*.npz'))
                logger.info(
                    'analyze %s: cache %s exists=%s, %d fit files',
                    name,
                    cache_path,
                    cache_path.exists(),
                    len(fits),
                )
                continue
            mb = torch.load(cache_path, weights_only=False)
            self._checkGrid(name, mb['grid'], cache_path)
            self._checkGrid(
                name, pd.read_csv(self.dataDir(name) / 'grid.csv', index_col='point'), 'grid.csv'
            )
            rows = []
            for block in mb['blocks']:
                for j, point in enumerate(block['points']):
                    batch1 = sliceBatch(block['batch'], j, j + 1)
                    refs = self.loadFits(name, point, batch1)
                    nuts = refs['NUTS'][0] if 'NUTS' in refs else None
                    point_diag = {}
                    if nuts is not None:
                        point_diag = {
                            'k_hat': self.kHat(name, block['MB0'], j, batch1),
                            'nuts_converged': refs['NUTS'][1]['nuts_converged'],
                        }
                    for method in METHODS:
                        p = self.proposal(block[method]).slice_b(j, j + 1)
                        row = (
                            self.rowBase(name, point, method)
                            | point_diag
                            | {'wall_s_batch': block[method]['wall_s']}
                        )
                        for key in ('accept_rate', 'map_z'):
                            if block[method][key] is not None:
                                row[key] = float(block[method][key][j])
                        rows.append(
                            row | self.agreement(name, p, nuts, batch1) | self.summaries(name, p)
                        )
                    for method, (p, diag) in refs.items():
                        row = self.rowBase(name, point, method) | diag
                        if p is not None:
                            agree = (
                                self.agreement(name, p, nuts, batch1) if method != 'NUTS' else {}
                            )
                            row |= agree | self.summaries(name, p)
                        rows.append(row)
            df = pd.DataFrame(rows).sort_values(['point', 'method'], kind='stable')
            if 'flagged' not in df:
                df['flagged'] = np.nan  # no NUTS fits yet
            df.to_csv(self.resultPath(f'_{name}.csv'), index=False)
            self.writeMd(name, df)
            frames[name] = df
        if frames:
            self.writeAgreementTable(frames)

    def _checkGrid(self, name: str, other: pd.DataFrame, source: str | Path) -> None:
        """Fits and caches are indexed by grid point, so they must come from the current grid."""
        grid = self.meta[name]['grid']
        same = (
            len(other) == len(grid)
            and (
                other[['ffx_family', 'sigma_family']].to_numpy()
                == grid[['ffx_family', 'sigma_family']].to_numpy()
            ).all()
            and np.allclose(
                other[['tau_beta', 'tau_sigma']].to_numpy(float),
                grid[['tau_beta', 'tau_sigma']].to_numpy(float),
            )
        )
        if not same:
            raise ValueError(
                f'{name}: {source} was produced from a different grid; re-run export / mb / NUTS'
            )

    def rowBase(self, name: str, point: int, method: str) -> dict:
        g = self.meta[name]['grid'].loc[point]
        return {
            'dataset': name,
            'point': point,
            'method': method,
            'ffx_family': g.ffx_family,
            'tau_beta': g.tau_beta,
            'sigma_family': g.sigma_family,
            'tau_sigma': g.tau_sigma,
            'nuts_subgrid': bool(g.nuts_subgrid),
        }

    @staticmethod
    def proposal(entry: dict) -> Proposal:
        return Proposal(entry['data'], has_sigma_eps=entry['has_sigma_eps'], d_corr=entry['d_corr'])

    def kHat(self, name: str, flow_entry: dict, j: int, batch1: dict) -> float:
        """PSIS k-hat of the flow pool at point j under the Api's IS correction (refine='is')."""
        lf = self.meta[name]['lf']
        data = {
            src: {k: v[j : j + 1].clone() for k, v in inner.items()}
            for src, inner in flow_entry['data'].items()
        }
        flow = Proposal(
            data, has_sigma_eps=flow_entry['has_sigma_eps'], d_corr=flow_entry['d_corr']
        )
        kw = dict(corr_prior=True, pareto=True, k_threshold=PSIS_K_THRESHOLD, likelihood_family=lf)
        if lf == 0:
            sampler = ImportanceSampler(batch1, marginal=True, rb_redraw=True, **kw)
        else:
            sampler = LaplaceImportanceSampler(batch1, **kw)
        return float(sampler(flow).is_results['pareto_k'][0])

    def loadFits(
        self, name: str, point: int, batch1: dict
    ) -> dict[str, tuple[Proposal | None, dict]]:
        """NUTS / ADVI fits of one grid point as Proposals padded to the Api batch layout
        (None for a failed ADVI fit)."""
        out = {}
        d_max, q_max, m_max = batch1['X'].shape[-1], batch1['Z'].shape[-1], batch1['X'].shape[1]
        for method, prefix in FIT_PREFIXES.items():
            path = self.dataDir(name) / 'fits' / f'test_{prefix}_{point:03d}.npz'
            if not path.exists():
                continue
            if path.stat().st_mtime < (self.dataDir(name) / 'test.npz').stat().st_mtime:
                # fits carry no priors, so one older than the export may come from another grid
                raise ValueError(
                    f'{path} is older than the grid export; re-run NUTS on this export'
                )
            with np.load(path, allow_pickle=True) as f:
                fit = dict(f)
            diag = {'duration_s': float(fit[f'{prefix}_duration'])}
            if bool(fit.get(f'{prefix}_failed', False)):
                out[method] = (None, diag | {'failed': True})
                continue
            t = lambda k: torch.as_tensor(fit[f'{prefix}_{k}'], dtype=torch.float32)
            ffx, sd = t('ffx').T, t('sigma_rfx').T  # (S, d), (S, q)
            S, d, q = ffx.shape[0], ffx.shape[1], sd.shape[1]
            parts = [
                torch.nn.functional.pad(ffx, (0, d_max - d)),
                torch.nn.functional.pad(sd, (0, q_max - q)),
            ]
            if self.meta[name]['lf'] == 0:
                parts.append(t('sigma_eps').T)  # (S, 1)
            rfx = t('rfx').permute(1, 2, 0)  # (q, m, S) -> (m, S, q)
            local = torch.zeros(m_max, S, q_max)
            local[: rfx.shape[0], :, :q] = rfx
            corr = torch.eye(q_max).repeat(S, 1, 1)  # (S, q_max, q_max)
            corr[:, :q, :q] = t('corr_rfx')[0]
            proposed = {
                'global': {'samples': torch.cat(parts, -1)[None]},
                'local': {'samples': local[None]},
            }
            p = Proposal(proposed, has_sigma_eps=self.meta[name]['lf'] == 0, corr_rfx=corr[None])
            if prefix == 'nuts':
                diag |= {
                    'rhat_max': float(np.nanmax(fit['nuts_rhat'])),
                    'ess_bulk_min': float(np.nanmin(fit['nuts_ess'])),
                    'divergences': int(fit['nuts_divergences'].sum()),
                    'treedepth_frac': float(np.mean(fit['nuts_max_treedepth'])),
                    'nuts_converged': nutsConverged(fit, CONVERGENCE_MODE),
                    'nuts_converged_strict': nutsConverged(fit, 'strict'),
                }
            out[method] = (p, diag)
        return out

    def draws(self, name: str, p: Proposal) -> dict[str, np.ndarray]:
        """Active parameters of a one-dataset Proposal, name -> (S,) draws."""
        meta = self.meta[name]
        d, q, m = meta['d'], meta['q'], meta['m']
        ffx, sd = p.ffx[0, :, :d].numpy(), p.sigma_rfx[0, :, :q].numpy()  # (S, d), (S, q)
        rfx = p.rfx[0, :m, :, :q].numpy()  # (m, S, q)
        out = {f'b_{n}': ffx[:, j] for j, n in enumerate(meta['ffx_names'])}
        out |= {f'sd_{n}': sd[:, j] for j, n in enumerate(meta['rfx_names'])}
        if p.has_sigma_eps:
            out['sigma_eps'] = p.sigma_eps[0].numpy()
        if q == 2:
            out['rho'] = p.corr_rfx[0, :, 1, 0].numpy()
        out |= {
            f'u{g}_{n}': rfx[g, :, j] for g in range(m) for j, n in enumerate(meta['rfx_names'])
        }
        return out

    def summaries(self, name: str, p: Proposal) -> dict[str, float]:
        draws = self.draws(name, p)
        qs = np.quantile(np.stack(list(draws.values())), list(QUANTILES.values()), axis=1)  # (3, P)
        return {
            f'{n}_{tag}': qs[k, i] for i, n in enumerate(draws) for k, tag in enumerate(QUANTILES)
        }

    def agreement(self, name: str, p: Proposal, nuts: Proposal | None, batch1: dict) -> dict:
        if nuts is None:
            return {}
        lf = self.meta[name]['lf']
        key = f'b_{DATASETS[name]["key"]}'
        a, b = self.draws(name, p)[key], self.draws(name, nuts)[key]
        width = lambda x: np.quantile(x, 0.95) - np.quantile(x, 0.05)
        key_z = abs(np.median(a) - np.median(b)) / b.std()
        near0 = lambda x: float((np.abs(x) < NEAR_ZERO).mean())
        width_ratio = width(a) / width(b)
        return {
            'r': computeCorr(p, nuts, batch1)[0],
            'sigma_ratio': computeSigmaRatio(p, nuts, batch1)[0],
            'rank_mad': computeRankMAD(p, nuts, batch1)[0],
            'delta_loo_nll': self.looNll(p, batch1, lf) - self.looNll(nuts, batch1, lf),
            'key_z': key_z,
            'key_width_ratio': width_ratio,
            'key_near0': near0(a),
            'key_near0_nuts': near0(b),
            'flagged': bool(
                key_z > KEY_Z_MAX or not WIDTH_RATIO_RANGE[0] <= width_ratio <= WIDTH_RATIO_RANGE[1]
            ),
        }

    @staticmethod
    def looNll(p: Proposal, batch1: dict, lf: int) -> float:
        pp = getPosteriorPredictive(p, batch1, likelihood_family=lf)
        return float(psisLooNLL(pp, batch1)[0][0])

    # --------------------------------------------------------------------------
    # reports

    def wallClock(self, name: str, df: pd.DataFrame) -> list[str]:
        lines = []
        for path in sorted(self.dataDir(name).glob('mb_*.pt')):
            mb = torch.load(path, weights_only=False)
            walls = {m: sum(b[m]['wall_s'] for b in mb['blocks']) for m in METHODS}
            lines.append(
                f'- metabeta on `{mb["device"]}`, {len(mb["grid"])} priors in {len(mb["blocks"])} batched calls, '
                f'{mb["n_samples"]} draws each, refinement chunk {mb["batch_size"] or "full batch"} '
                '(checkpoint load and warm-up excluded): '
                + ', '.join(f'{m} {w:.1f} s' for m, w in walls.items())
            )
        nuts = df[df.method == 'NUTS']
        if len(nuts):
            total = nuts.duration_s.sum()
            per_fit = nuts.duration_s.mean()
            lines.append(
                f'- NUTS ({NUTS_CHAINS} parallel chains, {NUTS_CHAINS} cores per fit), {len(nuts)} fits: '
                f'{total:.0f} s wall summed, {total * NUTS_CHAINS / 3600:.2f} core-hours '
                f'({per_fit:.1f} s per fit)'
            )
            n_grid = len(self.meta[name]['grid'])
            if len(nuts) < n_grid:
                lines.append(
                    f'- NUTS **extrapolated** to all {n_grid} grid points: '
                    f'{per_fit * n_grid * NUTS_CHAINS / 3600:.1f} core-hours '
                    f'({per_fit * n_grid / 3600:.1f} h if run serially)'
                )
            sub = nuts[nuts.nuts_subgrid]
            lines.append(
                f'- NUTS on the {len(sub)}-point sub-grid: '
                f'{sub.duration_s.sum() * NUTS_CHAINS / 3600:.2f} core-hours'
            )
        return lines

    def writeMd(self, name: str, df: pd.DataFrame) -> None:
        meta = self.meta[name]
        grid = meta['grid']
        sub = df[df.nuts_subgrid]
        n_sub = int(grid.nuts_subgrid.sum())
        missing = sorted(set(range(n_sub)) - set(sub[sub.method == 'NUTS'].point))
        parts = [
            f'# E2 prior sensitivity: {DATASETS[name]["title"]}',
            '## Setup (fixed 2026-09-24, before any NUTS result)\n\n'
            f'- formula `{self.formula(name)}` on the preprocessed data (model space, sd_y = 1); '
            f'd={meta["d"]}, q={meta["q"]}, m={meta["m"]}, n={meta["n"]}; key effect `{DATASETS[name]["key"]}`\n'
            f'- grid ({len(grid)} points, capped at the training hyper-prior range tau_beta <= '
            f'{TRAIN_MAX_TAU_FFX[meta["lf"]]}, tau_sigma <= {TRAIN_MAX_TAU_RFX[meta["lf"]]}; points above '
            f'the cap dropped): tau_beta in {np.round(grid.tau_beta.unique(), 3).tolist()} (every slope, '
            f'location 0; intercept scale {meta["tau_intercept"]:g}{self._interceptNote(name)}) x {list(FFX_FAMILIES)} x tau_sigma in '
            f'{grid.tau_sigma.unique().tolist()} x {list(SIGMA_FAMILIES)}; sigma_eps and LKJ eta at Bambi defaults\n'
            f'- NUTS sub-grid (points 0-{n_sub - 1}): all tau_beta x both ffx families x {meta["nuts_sd"]}\n'
            f'- disagreement flag: key |Δmedian| / sd_NUTS > {KEY_Z_MAX} or 90% width ratio outside '
            f'{list(WIDTH_RATIO_RANGE)}\n'
            f'- figures (fixed before the run): row 1 at {meta["nuts_sd"]} (main: half-Normal, appendix: '
            f'Exponential); row 2 at Normal(0, tau_beta = {meta["fig_tau_beta"]:.3g})\n'
            '- changes after the first metabeta run, before any NUTS result: draws per prior 1000 -> 4000; '
            'epil replaced by salamanders (no released Poisson submodel routes epil), with its key predictor '
            'only (all 11 predictors: IMH acceptance 0.02-0.03 in the pilot, not reported); grid capped at the '
            'training range (E1: extrapolation and IMH collapse beyond it)\n'
            f'- NUTS fits missing: {missing if missing else "none"}',
        ]
        metric_cols = [
            'r',
            'sigma_ratio',
            'rank_mad',
            'delta_loo_nll',
            'accept_rate',
            'key_z',
            'key_width_ratio',
        ]
        nuts_rows = sub[sub.method == 'NUTS']
        converged = nuts_rows[nuts_rows.nuts_converged.astype(bool)].point if len(nuts_rows) else []
        for label, mask in (
            (
                f'converged sub-grid points (primary; nutsConvergeMask {CONVERGENCE_MODE})',
                sub.point.isin(converged),
            ),
            ('all sub-grid points', sub.point >= 0),
        ):
            rows = []
            for method, g in sub[mask].groupby('method', sort=False):
                if method == 'NUTS':
                    continue
                cells = [medMad(g[c]) if c in g else 'NA' for c in metric_cols]
                rows.append([method, len(g)] + cells + [flagCount(g)])
            table = tabulate(
                rows, headers=['method', 'n', *metric_cols, 'flagged'], tablefmt='pipe'
            )
            parts.append(f'## Agreement with NUTS, {label} (median ± MAD)\n\n{table}')
        extra_fit = set(df[df.method == 'NUTS'].point) - set(sub.point)
        listed = (self.dataDir(name) / 'nuts_extra.txt').read_text().strip()
        listed = {int(i) for i in listed.split(',')} if listed else set()
        # the second run fitted NUTS on every grid point; the rest of the grid is reported apart
        # from the pre-specified sub-grid, and within it the low-acceptance points the pre-run rule
        # selected (MB acceptance < IMH_ACCEPT_WARN)
        for label, points in (
            (f'rest of the grid ({len(extra_fit)} points outside the sub-grid)', extra_fit),
            (
                f'low-acceptance points (MB acceptance < {IMH_ACCEPT_WARN:g}, outside the sub-grid; '
                f'listed {sorted(listed)}, NUTS fits missing {sorted(listed - extra_fit) or "none"})',
                listed & extra_fit,
            ),
        ):
            extra = df[df.point.isin(points) & (df.method != 'NUTS')]
            rows = [
                [method, len(g)]
                + [medMad(g[c]) if c in g else 'NA' for c in metric_cols]
                + [flagCount(g)]
                for method, g in extra.groupby('method', sort=False)
            ]
            table = tabulate(
                rows, headers=['method', 'n', *metric_cols, 'flagged'], tablefmt='pipe'
            )
            parts.append(f'## Agreement with NUTS, {label}\n\n' + (table if rows else 'none'))
        key = f'b_{DATASETS[name]["key"]}'
        flagged = df[
            df.point.isin(set(sub.point) | extra_fit) & df.flagged.fillna(False).astype(bool)
        ]
        if len(flagged):
            nuts = df[df.method == 'NUTS'].set_index('point')
            cols = ['point', 'method', 'ffx_family', 'tau_beta', 'sigma_family', 'tau_sigma']
            diag_cols = ['accept_rate', 'k_hat', 'key_near0', 'key_near0_nuts', 'nuts_converged']
            rows = [
                [
                    *r[cols],
                    r[f'{key}_q50'],
                    nuts.loc[r.point, f'{key}_q50'],
                    r.key_z,
                    r.key_width_ratio,
                ]
                + [r.get(c, np.nan) for c in diag_cols]
                for _, r in flagged.iterrows()
            ]
            table = tabulate(
                rows,
                headers=[*cols, 'median', 'median NUTS', 'z', 'width ratio', 'IMH acc.', 'k-hat',
                         f'share |b|<{NEAR_ZERO:g}', f'NUTS share |b|<{NEAR_ZERO:g}', 'NUTS converged'],
                tablefmt='pipe',
                floatfmt='.3g',
            )  # fmt: skip
            parts.append(f'## Flagged points\n\n{table}')
        nuts = sub[sub.method == 'NUTS']
        if len(nuts):
            bad = nuts[~nuts.nuts_converged.astype(bool)]
            parts.append(
                f'## NUTS diagnostics\n\nconverged ({CONVERGENCE_MODE}) at {len(nuts) - len(bad)}/{len(nuts)} '
                f'sub-grid points (strict: {int(nuts.nuts_converged_strict.sum())}/{len(nuts)}); max R-hat '
                f'{nuts.rhat_max.max():.3f}, min bulk ESS {nuts.ess_bulk_min.min():.0f}; not converged: '
                + (
                    str(
                        bad[['point', 'rhat_max', 'divergences', 'ess_bulk_min']]
                        .round(3)
                        .to_dict('records')
                    )
                    if len(bad)
                    else 'none'
                )
            )
            parts.append(self.diagnosticCrossTab(sub))
        parts.append('## Wall-clock\n\n' + '\n'.join(self.wallClock(name, df)))
        path = self.resultPath(f'_{name}.md')
        path.write_text('\n\n'.join(parts) + '\n')
        logger.info('saved %s', path)

    @staticmethod
    def diagnosticCrossTab(sub: pd.DataFrame) -> str:
        """MB disagreement flag vs the built-in diagnostics, over all sub-grid points."""
        g = sub[(sub.method == 'MB') & sub.flagged.notna()]
        flag = g.flagged.astype(bool)
        fired = (g.accept_rate < FIRE_ACCEPT) | (g.k_hat > FIRE_KHAT)
        counts = {
            'flagged & fired': int((flag & fired).sum()),
            'flagged & silent': int((flag & ~fired).sum()),
            'unflagged & fired': int((~flag & fired).sum()),
            'unflagged & silent': int((~flag & ~fired).sum()),
        }
        return (
            f'## MB disagreement flag vs built-in diagnostics (all {len(g)} sub-grid points; fired = '
            f'IMH acceptance < {FIRE_ACCEPT:g} or k-hat > {FIRE_KHAT:g})\n\n'
            + tabulate([list(counts.values())], headers=list(counts), tablefmt='pipe')
        )

    def _interceptNote(self, name: str) -> str:
        lf = self.meta[name]['lf']
        default = float(bambiDefaultPriors(1, 1, lf)['tau_ffx'][0])
        if default <= TRAIN_MAX_TAU_FFX[lf]:
            return ', the Bambi default'
        return f', the Bambi default {default:g} capped at the training maximum'

    def writeAgreementTable(self, frames: dict[str, pd.DataFrame]) -> None:
        cols = ['r', 'sigma_ratio', 'rank_mad', 'delta_loo_nll', 'accept_rate']
        md_rows, tex_rows, reliable = [], [], []
        for name, df in frames.items():
            nuts = df[df.nuts_subgrid & (df.method == 'NUTS')]
            conv = set(nuts[nuts.nuts_converged.astype(bool)].point)
            reliable.append(
                f'- {name}: NUTS converged ({CONVERGENCE_MODE}) at {len(conv)}/{len(nuts)} sub-grid points '
                f'(strict: {int(nuts.nuts_converged_strict.sum())})'
            )
            sub = df[df.nuts_subgrid & (df.method != 'NUTS')]
            for method, g_all in sub.groupby('method', sort=False):
                g = g_all[g_all.point.isin(conv)]
                cells = [medMad(g[c]) if c in g and g[c].notna().any() else 'NA' for c in cols]
                flag, flag_all = flagCount(g), flagCount(g_all)
                md_rows.append([name, method] + cells + [flag, flag_all])
                tex_cells = [c.replace(' ± ', r' \pm ') for c in cells]
                tex_rows.append(
                    rf'      \texttt{{{name}}} & \texttt{{{method}}} & '
                    + ' & '.join(f'${c}$' if c != 'NA' else r'\textrm{NA}' for c in tex_cells)
                    + rf' & ${flag}$ & ${flag_all}$ \\'
                )
        headers = [
            'dataset',
            'method',
            'r',
            'σ-ratio',
            'rank-MAD',
            'ΔLOO-NLL',
            'IMH acc.',
            'flagged (converged)',
            'flagged (all)',
        ]
        md = (
            '# E2 agreement with NUTS (median ± MAD over converged sub-grid points; '
            f'nutsConvergeMask {CONVERGENCE_MODE})\n\n'
        )
        md += (
            tabulate(md_rows, headers=headers, tablefmt='pipe')
            + '\n\n'
            + '\n'.join(reliable)
            + '\n'
        )
        self.resultPath('_agreement.md').write_text(md)
        tex = [
            f'% entries: median ± MAD over converged NUTS sub-grid points (nutsConvergeMask {CONVERGENCE_MODE})',
            r'\begin{tabular}{ll|ccccccc}',
            r'    \toprule',
            r'    dataset & model & $r$ & $\sigma\text{-ratio}$ & $\mathrm{rank\text{-}MAD}$ & '
            r'$\Delta\mathrm{LOO\text{-}NLL}$ & acc. & flagged (conv.) & flagged (all) \\',
            r'    \midrule',
            *tex_rows,
            r'    \bottomrule',
            r'\end{tabular}',
            '',
        ]
        self.resultPath('_agreement.tex').write_text('\n'.join(tex))
        logger.info('saved %s.md/.tex', self.resultPath('_agreement'))

    # --------------------------------------------------------------------------
    # evidence

    def evidence(self) -> None:
        """IS log p(D) (flow pool, all points) vs bridge sampling on NUTS draws (sub-grid)."""
        sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'posthoc'))
        from evidence import Target, bridge, isEvidence, toDouble

        S = self.cfg.n_samples
        for name in [n for n in self.data if self.meta[n]['lf'] == 0]:
            if self.cfg.dry_run:
                logger.info(
                    'evidence %s: IS on %d flow pools of %d, bridge on the NUTS fits',
                    name,
                    len(self.meta[name]['grid']),
                    S,
                )
                continue
            cache_path = self.dataDir(name) / f'mb_{self.cfg.device}.pt'
            mb = torch.load(cache_path, weights_only=False)
            self._checkGrid(name, mb['grid'], cache_path)
            gen = torch.Generator().manual_seed(self.cfg.seed)
            rows = []
            for block in mb['blocks']:
                flow = self.proposal(block['MB0'])
                for j, point in enumerate(block['points']):
                    batch64 = toDouble(sliceBatch(block['batch'], j, j + 1))
                    ev = isEvidence(flow.slice_b(j, j + 1), batch64, [S])
                    row = self.rowBase(name, point, 'MB0') | {
                        'logev_is': ev[f'logev_is_s{S}'],
                        'psis_k': ev[f'k_s{S}'],
                        'is_eff': ev[f'eff_s{S}'],
                    }
                    path = self.dataDir(name) / 'fits' / f'test_nuts_{point:03d}.npz'
                    if path.exists():
                        with np.load(path, allow_pickle=True) as f:
                            fit = {
                                k: torch.as_tensor(f[k])
                                for k in (
                                    'nuts_ffx',
                                    'nuts_sigma_rfx',
                                    'nuts_sigma_eps',
                                    'nuts_corr_rfx',
                                )
                            }
                        target = Target(batch64, block['MB0']['d_corr'])
                        u = target.fromConstrained(
                            fit['nuts_ffx'].T,
                            fit['nuts_sigma_rfx'].T,
                            fit['nuts_sigma_eps'][0],
                            fit['nuts_corr_rfx'][0],
                        )
                        br = bridge(target.logProb, u, self.cfg.n_bridge, gen)
                        row |= {
                            'logev_bridge': br['log_ev'],
                            'logev_bridge_swap': br['log_ev_swap'],
                        }
                    rows.append(row)
            df = pd.DataFrame(rows).sort_values('point')
            path = self.resultPath(f'_evidence_{name}.csv')
            df.to_csv(path, index=False)
            logger.info('saved %s', path)

    # --------------------------------------------------------------------------
    # plot

    def plot(self) -> None:
        names = list(self.data)
        if self.cfg.dry_run:
            logger.info(
                'plot: %s from %s',
                Path(self.cfg.fig_dir) / self.resultPath('.pdf').name,
                [self.resultPath(f'_{n}.csv').name for n in names],
            )
            return
        plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14})
        for suffix, sd_family in FIG_SD_FAMILY.items():
            self._save(self._figure(names, sd_family), suffix)
        # main text: key-effect row only; the 2 x 3 figures go to the appendix
        self._save(self._figure(names, 'halfnormal', rows=1), '_main')
        normal = [n for n in names if self.meta[n]['lf'] == 0]
        if normal and all(self.resultPath(f'_evidence_{n}.csv').exists() for n in normal):
            self.plotEvidence()

    def _figure(self, names: list[str], sd_family: str, rows: int = 2):
        fig, axes = plt.subplots(
            rows, len(names), figsize=(4.4 * len(names), 3.8 * rows), squeeze=False
        )
        for c, name in enumerate(names):
            df = pd.read_csv(self.resultPath(f'_{name}.csv'))
            meta = self.meta[name]
            key = f'b_{DATASETS[name]["key"]}'
            sd_prior = (sd_family, meta['nuts_sd'][sd_family])
            self._panelTauBeta(axes[0, c], df, key, sd_prior)
            if rows == 2:
                self._panelTauSigma(axes[1, c], df, 'sd_Intercept', meta['fig_tau_beta'])
                axes[1, c].set_ylabel(r'$\sigma_{\mathrm{Intercept}}$')
            sd_label = {'halfnormal': 'HN', 'exponential': 'Exp'}[sd_family]
            axes[0, c].set_title(f'{DATASETS[name]["title"]}, SD prior {sd_label}({sd_prior[1]:g})')
            axes[0, c].set_ylabel(DATASETS[name]['key_label'])
        axes[0, 0].legend(
            [Line2D([], [], color=C_MB, ls='-'), Line2D([], [], color=C_NUTS, marker='o', ls=''),
             Line2D([], [], color=C_UNCONV, marker='o', mfc='white', ls='')],
            ['MB, Normal', 'NUTS, Normal', 'NUTS, not converged'],
            fontsize=10, loc='best',
        )  # fmt: skip
        if rows == 2:
            sd_ls = {'halfnormal': '-', 'halfstudent': '--', 'exponential': ':'}
            axes[1, 0].legend(
                [Line2D([], [], color=C_MB, ls=ls) for ls in sd_ls.values()]
                + [Line2D([], [], color=C_NUTS, marker=mk, ls='') for mk in SD_MARKERS.values()],
                ['MB, half-Normal', 'MB, half-Student-t', 'MB, Exponential', 'NUTS, half-Normal', 'NUTS, Exponential'],
                fontsize=10, loc='best',
            )  # fmt: skip
        fig.tight_layout(rect=(0, 0, 0.93, 1))
        cax = fig.add_axes((0.945, 0.25 if rows == 2 else 0.18, 0.012, 0.5 if rows == 2 else 0.68))
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=ACCEPT_NORM, cmap=ACCEPT_CMAP), cax=cax)
        cbar.set_label('MB: IMH acceptance')
        cbar.ax.axhline(IMH_ACCEPT_WARN, color='0.2', lw=1.2)  # the Api's warning threshold
        return fig

    @staticmethod
    def _acceptMarkers(ax, x: pd.Series, y: pd.Series, accept: pd.Series, marker: str) -> None:
        ax.scatter(x, y, c=accept, cmap=ACCEPT_CMAP, norm=ACCEPT_NORM, marker=marker, s=30,
                   edgecolors=C_MB, linewidths=0.6, zorder=3)  # fmt: skip

    @staticmethod
    def _nutsDots(ax, nu: pd.DataFrame, x: pd.Series, key: str, marker: str, mfc: str) -> None:
        """NUTS median and 90% interval; points failing nutsConvergeMask hollow grey."""
        for conv, color, face in ((True, C_NUTS, mfc), (False, C_UNCONV, 'white')):
            sel = nu.nuts_converged.astype(bool) == conv
            n = nu[sel]
            yerr = [n[f'{key}_q50'] - n[f'{key}_q05'], n[f'{key}_q95'] - n[f'{key}_q50']]
            ax.errorbar(
                x[sel], n[f'{key}_q50'], yerr=yerr, fmt=marker, color=color, mfc=face, ms=4.5, lw=1
            )

    def _panelTauBeta(self, ax, df: pd.DataFrame, key: str, sd_prior: tuple[str, float]) -> None:
        # Normal slope priors only; the Student-t results are reported in the addendum tables
        sl = df[
            (df.ffx_family == 'normal')
            & (df.sigma_family == sd_prior[0])
            & np.isclose(df.tau_sigma, sd_prior[1])
        ]
        mb = sl[sl.method == 'MB'].sort_values('tau_beta')
        ax.plot(mb.tau_beta, mb[f'{key}_q50'], color=C_MB, lw=1.6)
        self._acceptMarkers(ax, mb.tau_beta, mb[f'{key}_q50'], mb.accept_rate, 'o')
        ax.fill_between(
            mb.tau_beta, mb[f'{key}_q05'], mb[f'{key}_q95'], color=C_MB, alpha=0.13, lw=0
        )
        nu = sl[sl.method == 'NUTS']
        self._nutsDots(ax, nu, nu.tau_beta * 0.96, key, 'o', C_NUTS)
        self._axis(ax, df.tau_beta, r'$\tau_\beta$')

    def _panelTauSigma(self, ax, df: pd.DataFrame, key: str, tau_beta: float) -> None:
        sl = df[(df.ffx_family == 'normal') & np.isclose(df.tau_beta, tau_beta)]
        for fam, ls in (('halfnormal', '-'), ('halfstudent', '--'), ('exponential', ':')):
            mb = sl[(sl.method == 'MB') & (sl.sigma_family == fam)].sort_values('tau_sigma')
            ax.plot(mb.tau_sigma, mb[f'{key}_q50'], color=C_MB, ls=ls, lw=1.6)
            self._acceptMarkers(
                ax, mb.tau_sigma, mb[f'{key}_q50'], mb.accept_rate, SD_MARKERS.get(fam, 'v')
            )
            ax.fill_between(
                mb.tau_sigma, mb[f'{key}_q05'], mb[f'{key}_q95'], color=C_MB, alpha=0.10, lw=0
            )
        for fam, marker in SD_MARKERS.items():
            nu = sl[(sl.method == 'NUTS') & (sl.sigma_family == fam)]
            self._nutsDots(ax, nu, nu.tau_sigma, key, marker, C_NUTS)
        self._axis(ax, df.tau_sigma, r'$\tau_\sigma$')

    @staticmethod
    def _axis(ax, x: pd.Series, xlabel: str) -> None:
        """Log x-axis spanning this dataset's (capped) grid values."""
        ax.set_xscale('log')
        ax.set_xlim(x.min() / 1.3, x.max() * 1.3)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if x.nunique() <= 4:  # tau_sigma: label the grid values themselves
            ax.set_xticks(sorted(x.unique()), [f'{v:g}' for v in sorted(x.unique())])
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    def plotEvidence(self) -> None:
        names = [n for n in self.data if self.meta[n]['lf'] == 0]
        fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.0), squeeze=False)
        for ax, name in zip(axes[0], names):
            ev = pd.read_csv(self.resultPath(f'_evidence_{name}.csv'))
            meta = self.meta[name]
            ev = ev[ev.ffx_family == 'normal']
            ts, tb = meta['nuts_sd']['halfnormal'], meta['fig_tau_beta']
            ref = ev[
                (ev.sigma_family == 'halfnormal')
                & np.isclose(ev.tau_sigma, ts)
                & np.isclose(ev.tau_beta, tb)
            ]
            for (sfam, stau), color in zip(meta['nuts_sd'].items(), (C_MB, PALETTE[0])):
                sl = ev[(ev.sigma_family == sfam) & np.isclose(ev.tau_sigma, stau)]
                sl = sl.sort_values('tau_beta')
                ax.plot(sl.tau_beta, sl.logev_is - ref.logev_is.item(), color=color, lw=1.6,
                        label=f'MB IS, normal, {sfam}({stau:g})')  # fmt: skip
                ax.plot(sl.tau_beta, sl.logev_bridge - ref.logev_bridge.item(), marker='o',
                        color=C_NUTS, ls='', ms=4.5)  # fmt: skip
            ax.plot(
                [],
                [],
                marker='o',
                color=C_NUTS,
                ls='',
                ms=4.5,
                label='bridge sampling on NUTS draws',
            )
            ax.axhline(0.0, color='0.5', lw=0.8)
            self._axis(ax, ev.tau_beta, r'$\tau_\beta$')
            ax.set_ylabel(rf'$\ln \mathrm{{BF}}$ vs Normal({tb:.3g}), HN({ts:g})')
            ax.set_title(DATASETS[name]['title'])
            ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, '_bf')

    def _save(self, fig, suffix: str) -> None:
        path = self.resultPath(f'{suffix}.pdf')
        for out in (Path(self.cfg.fig_dir) / path.name, path):
            fig.savefig(out, bbox_inches='tight', pad_inches=0.15, dpi=VECTOR_RASTER_DPI)
            logger.info('saved %s', out)
        plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    study = PriorSensitivity(cfg)
    study.logPlan()
    stages = {
        'export': study.export,
        'mb': study.sampleMB,
        'analyze': study.analyze,
        'evidence': study.evidence,
        'plot': study.plot,
    }
    for stage in cfg.stages:
        stages[stage]()
