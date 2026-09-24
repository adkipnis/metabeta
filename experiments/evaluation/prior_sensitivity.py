"""
experiments/evaluation/prior_sensitivity.py — E2: zero-shot prior sensitivity analysis on real data.

Three held-out real datasets (one per likelihood family: sleepstudy, cbpp, salamanders) are analysed under a 288-point grid of
analyst priors with the released checkpoints (eval only).  Each (fixed-effect family, rfx-SD
family) combination is one batched Api.sample call of 48 priors, because the router takes one
family combination per batch.  NUTS on a 48-point sub-grid is the reference.

Everything below was fixed on 2026-09-24, before any NUTS result.  Two changes followed the first
metabeta run (sleepstudy, cbpp): the draw count went from 1000 to 4000, because IMH acceptance fell
below 0.1 at tau_beta <= 0.076; and epil (the plan's Poisson set) was replaced by salamanders,
because every released Poisson submodel needs >= 5 observations per group and epil has 4.

Model space.  Priors act on the preprocessed data that both the Api and NUTS see: sleepstudy y
and Days are standardised (sd_y = 1, so the Api's sd_y rescaling is the identity), cbpp keeps its
0/1 period dummies, salamanders its covariates and 0/1 dummies as preprocessed.  NUTS receives the
identical prior arrays (see `prior`), so no unit conversion sits between the two.

Grid (288 points per dataset):
  tau_beta   12 log-spaced values 0.05 ... 5.0, one scale for every slope, location 0; the
             intercept keeps its Bambi default scale (2.5 / 1.5 / 2.5 for Normal / Bernoulli /
             Poisson): its natural size is the baseline rate, not an effect size, so shrinking it
             with the slopes would confound the key-effect curve with a baseline-rate conflict
  ffx family Normal, Student-t(5)
  tau_sigma  0.25, 1, 2.5, 5 (every rfx SD)
  SD family  half-Normal, half-Student-t(5), Exponential (mean tau_sigma)
  sigma_eps and LKJ eta at the Bambi defaults (utils/priors.bambiDefaultPriors)
NUTS sub-grid = grid points 0-47: all tau_beta x both ffx families x {HN(2.5), Exp(1)}.
Points whose varied scales (slope tau_beta, tau_sigma) lie outside the training hyper-prior
range (simulation/prior.hypersample) are flagged `in_support = False`, never dropped; whether the
fixed intercept scale lies inside is stated once per dataset in the .md.

NUTS is a reliable reference at a point when R-hat <= 1.01 and there are 0 divergences; the
agreement tables report that subset separately and list unreliable points, which count neither
for nor against MB.

Disagreement flag (key effect, per NUTS point): |median_MB - median_NUTS| / sd_NUTS > 0.25, or
90% interval width ratio MB / NUTS outside [0.8, 1.25].

Figures (slice choice fixed before the run): row 1 = key effect vs tau_beta at SD prior HN(2.5)
in the main figure (prior_sensitivity.pdf) and Exp(1) in the appendix backup
(prior_sensitivity_exp1.pdf); row 2 (both figures) = random-intercept SD vs
tau_sigma at Normal(0, tau_beta = 2.16), the grid value nearest the Bambi default 2.5.
Lines = MB (flow + IMH, the paper default), dots = NUTS.

Stages (--stages, run in the order given):
  export    the grid as a fit.py batch: outputs/data/e2-{ds}/test.npz + config.yaml (no model)
  mb        MB0 (raw flow) and MB (flow + IMH) on all 288 points, cached with per-batch
            wall-clock to outputs/data/e2-{ds}/mb_{device}.pt (loads the released checkpoint)
  analyze   MB cache + NUTS (and ADVI, if present) fits -> results/prior_sensitivity_{ds}.csv/.md
            and the appendix table prior_sensitivity_agreement.md/.tex
  evidence  Normal datasets only: IS log p(D) from the flow pool on all 288 points vs Meng-Wong
            bridge sampling on the NUTS draws (experiments/posthoc/evidence.py)
  plot      prior_sensitivity.pdf (+ prior_sensitivity_bf.pdf after evidence)

Usage (from repo root):
    uv run python experiments/evaluation/prior_sensitivity.py --stages export mb analyze --dry_run
    uv run python experiments/evaluation/prior_sensitivity.py --stages export
    sbatch scripts/fit-nuts-prior-grid.sh --dataset sleep          # cluster, 48 NUTS jobs per dataset
    uv run python experiments/evaluation/prior_sensitivity.py --stages mb analyze evidence plot
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
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate

from metabeta.evaluation.predictive import getPosteriorPredictive, psisLooNLL
from metabeta.utils.api import coercePriors, parseFormula, resolveLikelihoodFamily
from metabeta.utils.constants import FFX_FAMILIES, LIKELIHOOD_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.dataloader import sliceBatch, toDevice
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

# formula = every preprocessed column as a fixed effect, rfx structure as in the plan (E2)
DATASETS = {
    'sleep': {
        'file': 'sleep__grp_group',
        'formula': 'y ~ days + (1 + days | group)',
        'key': 'days',
        'title': 'sleepstudy (Normal)',
        'key_label': r'$\beta_{\mathrm{Days}}$',
    },
    'cbpp': {
        'file': 'cbpp__grp_group',
        'formula': 'y ~ period_2 + period_3 + period_4 + (1 | group)',
        'key': 'period_4',
        'title': 'cbpp (Bernoulli)',
        'key_label': r'$\beta_{\mathrm{period\,4}}$',
    },
    # replaces epil, which no released Poisson submodel routes (4 observations per group < 5)
    'salamanders': {
        'file': 'salamanders__grp_group',
        'formula': 'y ~ cover + sample + dop + wtemp + mined_yes + spp_DES-L + spp_DF + spp_DM'
        ' + spp_EC-L + spp_GP + spp_PR + (1 | group)',
        'key': 'mined_yes',
        'title': 'salamanders (Poisson)',
        'key_label': r'$\beta_{\mathrm{mined}}$',
    },
}

TAU_BETA = np.geomspace(0.05, 5.0, 12)
TAU_SIGMA = (0.25, 1.0, 2.5, 5.0)
NUTS_SD_PRIORS = (('halfnormal', 2.5), ('exponential', 1.0))
# largest tau_ffx / tau_rfx the training hyper-prior draws (simulation/prior.py, hypersample)
TRAIN_MAX_TAU_FFX = {0: 4.0, 1: 3.0, 2: 1.5}
TRAIN_MAX_TAU_RFX = {0: 5.0, 1: 2.0, 2: 1.0}

METHODS = {'MB0': False, 'MB': True}  # label -> Api.sample(refine=...)
FIT_PREFIXES = {'NUTS': 'nuts', 'ADVI': 'advi'}
QUANTILES = {'q05': 0.05, 'q50': 0.5, 'q95': 0.95}
NUTS_CHAINS = 4  # fit.py default; chains run in parallel, one core each

KEY_Z_MAX = 0.25
WIDTH_RATIO_RANGE = (0.8, 1.25)
FIG_SD_PRIORS = {  # figure stem -> row-1 SD prior
    'prior_sensitivity': ('halfnormal', 2.5),
    'prior_sensitivity_exp1': ('exponential', 1.0),
}
FIG_TAU_BETA = TAU_BETA[9]  # 2.16, nearest grid value to the Bambi default 2.5
BF_REFERENCE = ('normal', 'halfnormal', 2.5, FIG_TAU_BETA)

SD_MARKERS = {'halfnormal': 'o', 'exponential': 'D'}  # NUTS SD priors in figure row 2
FIG_DIR = Path.home() / 'LaTeX' / 'metabeta-iclr' / 'figures'
C_MB, C_NUTS = PALETTE[4], PALETTE[3]  # runtimes figure colours of MB and NUTS


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', nargs='+', required=True, choices=['export', 'mb', 'analyze', 'evidence', 'plot'])
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


def buildGrid() -> pd.DataFrame:
    """The 288 priors; the 48 NUTS points come first so a SLURM array 0-47 covers them."""
    rows = [
        {'ffx_family': f, 'sigma_family': s, 'tau_sigma': ts, 'tau_beta': tb}
        for f in FFX_FAMILIES
        for s in SIGMA_FAMILIES
        for ts in TAU_SIGMA
        for tb in TAU_BETA
    ]
    grid = pd.DataFrame(rows)
    grid['nuts_subgrid'] = [
        (s, ts) in NUTS_SD_PRIORS for s, ts in zip(grid.sigma_family, grid.tau_sigma)
    ]
    grid = grid.sort_values('nuts_subgrid', ascending=False, kind='stable').reset_index(drop=True)
    grid.index.name = 'point'
    return grid


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
        self.grid = buildGrid()
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
        formula = parseFormula(spec['formula'])
        if tuple(formula.fixed_terms) != tuple(c.lower() for c in columns):  # parser lowercases
            raise ValueError(f'{name}: formula terms {formula.fixed_terms} != columns {columns}')
        if 'sd_y' in data:
            raise ValueError(f'{name}: the grid is defined for sd_y = 1 (no sd_y key)')
        rfx_names = ['Intercept' if t == '1' else t for t in formula.random_terms]
        lf = resolveLikelihoodFamily(None, str(data['y_type']))
        self.data[name] = data
        self.meta[name] = {
            'lf': lf,
            'd': len(columns) + 1,
            'q': len(rfx_names),
            'm': int(data['m']),
            'n': int(data['n']),
            'ffx_names': ['Intercept', *columns],
            'rfx_names': rfx_names,
            'tau_intercept': float(bambiDefaultPriors(1, 1, lf)['tau_ffx'][0]),
            'in_support': (self.grid.tau_beta <= TRAIN_MAX_TAU_FFX[lf])
            & (self.grid.tau_sigma <= TRAIN_MAX_TAU_RFX[lf]),
        }

    def dataDir(self, name: str) -> Path:
        return DATA_DIR / f'e2-{name}'

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

    def blocks(self) -> list[pd.DataFrame]:
        """One block of 48 points per (ffx family, SD family): one batched Api call each."""
        return [b for _, b in self.grid.groupby(['ffx_family', 'sigma_family'], sort=False)]

    def logPlan(self) -> None:
        for name, meta in self.meta.items():
            sub = self.grid.nuts_subgrid
            logger.info(
                '%s: %s | lf=%d d=%d q=%d m=%d n=%d | %d/288 points in training support '
                '(%d/48 of the NUTS sub-grid)',
                name, DATASETS[name]['formula'], meta['lf'], meta['d'], meta['q'], meta['m'],
                meta['n'], meta['in_support'].sum(), meta['in_support'][sub].sum(),
            )  # fmt: skip
        logger.info('tau_beta grid: %s', np.round(TAU_BETA, 4).tolist())
        logger.info('blocks: %s', [len(b) for b in self.blocks()])

    # --------------------------------------------------------------------------
    # export

    def export(self) -> None:
        """Write the grid as a fit.py batch (identical data, one prior per row)."""
        for name, data in self.data.items():
            meta = self.meta[name]
            n, d, q, m = meta['n'], meta['d'], meta['q'], meta['m']
            base = {
                'X': np.concatenate([np.ones((n, 1)), data['X']], axis=1),  # (n, d)
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
            priors = [self.prior(name, point) for _, point in self.grid.iterrows()]
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
            self.grid.to_csv(out / 'grid.csv')

    # --------------------------------------------------------------------------
    # mb

    def sampleMB(self) -> None:
        from metabeta.models.api import Api

        S = self.cfg.n_samples
        for name, data in self.data.items():
            meta = self.meta[name]
            formula = DATASETS[name]['formula']
            api_priors = [
                {f'p{i:03d}': self.prior(name, point) for i, point in block.iterrows()}
                for block in self.blocks()
            ]
            if self.cfg.dry_run:
                logger.info(
                    'mb %s: %d calls x 48 priors x %d draws x %s',
                    name,
                    len(api_priors),
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
            for block, priors in zip(self.blocks(), api_priors):
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
            out = self.dataDir(name) / f'mb_{self.cfg.device}.pt'
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'device': self.cfg.device,
                    'n_samples': S,
                    'batch_size': self.cfg.batch_size,
                    'blocks': blocks,
                },
                out,
            )
            logger.info('saved %s', out)

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
            rows = []
            for block in mb['blocks']:
                for j, point in enumerate(block['points']):
                    batch1 = sliceBatch(block['batch'], j, j + 1)
                    refs = self.loadFits(name, point, batch1)
                    nuts = refs['NUTS'][0] if 'NUTS' in refs else None
                    for method in METHODS:
                        p = self.proposal(block[method]).slice_b(j, j + 1)
                        row = self.rowBase(name, point, method) | {
                            'wall_s_batch': block[method]['wall_s']
                        }
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
            df.to_csv(RESULTS_DIR / f'prior_sensitivity_{name}.csv', index=False)
            self.writeMd(name, df)
            frames[name] = df
        if frames:
            self.writeAgreementTable(frames)

    def rowBase(self, name: str, point: int, method: str) -> dict:
        g = self.grid.loc[point]
        return {
            'dataset': name,
            'point': point,
            'method': method,
            'ffx_family': g.ffx_family,
            'tau_beta': g.tau_beta,
            'sigma_family': g.sigma_family,
            'tau_sigma': g.tau_sigma,
            'nuts_subgrid': bool(g.nuts_subgrid),
            'in_support': bool(self.meta[name]['in_support'][point]),
        }

    @staticmethod
    def proposal(entry: dict) -> Proposal:
        return Proposal(entry['data'], has_sigma_eps=entry['has_sigma_eps'], d_corr=entry['d_corr'])

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
            with np.load(path, allow_pickle=True) as f:
                fit = dict(f)
            if prefix == 'advi' and bool(fit['advi_failed']):
                out[method] = (None, {'advi_failed': True})
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
            diag = {'duration_s': float(fit[f'{prefix}_duration'])}
            if prefix == 'nuts':
                diag |= {
                    'rhat_max': float(np.nanmax(fit['nuts_rhat'])),
                    'ess_bulk_min': float(np.nanmin(fit['nuts_ess'])),
                    'divergences': int(fit['nuts_divergences'].sum()),
                    'treedepth_frac': float(np.mean(fit['nuts_max_treedepth'])),
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
        width_ratio = width(a) / width(b)
        return {
            'r': computeCorr(p, nuts, batch1)[0],
            'sigma_ratio': computeSigmaRatio(p, nuts, batch1)[0],
            'rank_mad': computeRankMAD(p, nuts, batch1)[0],
            'delta_loo_nll': self.looNll(p, batch1, lf) - self.looNll(nuts, batch1, lf),
            'key_z': key_z,
            'key_width_ratio': width_ratio,
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
                f'- metabeta on `{mb["device"]}`, 288 priors in {len(mb["blocks"])} batched calls, '
                f'{mb["n_samples"]} draws each, refinement chunk {mb["batch_size"] or 48} '
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
            lines.append(
                f'- NUTS **extrapolated** to 288 fits: {per_fit * 288 * NUTS_CHAINS / 3600:.1f} core-hours '
                f'({per_fit * 288 / 3600:.1f} h if run serially)'
            )
        return lines

    def writeMd(self, name: str, df: pd.DataFrame) -> None:
        meta = self.meta[name]
        sub = df[df.nuts_subgrid]
        missing = sorted(set(range(48)) - set(sub[sub.method == 'NUTS'].point))
        parts = [
            f'# E2 prior sensitivity: {DATASETS[name]["title"]}',
            '## Setup (fixed 2026-09-24, before any run)\n\n'
            f'- formula `{DATASETS[name]["formula"]}` on the preprocessed data (model space, sd_y = 1); '
            f'd={meta["d"]}, q={meta["q"]}, m={meta["m"]}, n={meta["n"]}; key effect `{DATASETS[name]["key"]}`\n'
            f'- grid: tau_beta in {np.round(TAU_BETA, 3).tolist()} (every slope, location 0; intercept '
            f'at its Bambi default scale {meta["tau_intercept"]:g}) x {list(FFX_FAMILIES)} x tau_sigma in {list(TAU_SIGMA)} x {list(SIGMA_FAMILIES)}; '
            'sigma_eps and LKJ eta at Bambi defaults\n'
            f'- NUTS sub-grid (points 0-47): all tau_beta x both ffx families x {list(NUTS_SD_PRIORS)}\n'
            f'- training support: tau_beta <= {TRAIN_MAX_TAU_FFX[meta["lf"]]}, tau_sigma <= '
            f'{TRAIN_MAX_TAU_RFX[meta["lf"]]}; {int(meta["in_support"].sum())}/288 points inside, '
            f'{int(meta["in_support"][:48].sum())}/48 of the sub-grid; the intercept scale '
            f'{meta["tau_intercept"]:g} is {"inside" if meta["tau_intercept"] <= TRAIN_MAX_TAU_FFX[meta["lf"]] else "OUTSIDE"} '
            'the training range at every point\n'
            f'- disagreement flag: key |Δmedian| / sd_NUTS > {KEY_Z_MAX} or 90% width ratio outside '
            f'{list(WIDTH_RATIO_RANGE)}\n'
            f'- figure row-1 SD prior (fixed before the run): {FIG_SD_PRIORS}\n'
            '- changes after the first metabeta run, before any NUTS result: draws per prior 1000 -> 4000; '
            'epil replaced by salamanders (no released Poisson submodel routes epil)\n'
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
        reliable = (
            nuts_rows[(nuts_rows.rhat_max <= 1.01) & (nuts_rows.divergences == 0)].point
            if len(nuts_rows)
            else []
        )
        for label, mask in (
            ('all sub-grid points', sub.point >= 0),
            ('in training support', sub.in_support),
            ('NUTS reliable (R-hat <= 1.01, 0 divergences)', sub.point.isin(reliable)),
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
        key = f'b_{DATASETS[name]["key"]}'
        flagged = sub[sub.flagged.fillna(False).astype(bool)]
        if len(flagged):
            nuts = sub[sub.method == 'NUTS'].set_index('point')
            cols = [
                'point',
                'method',
                'ffx_family',
                'tau_beta',
                'sigma_family',
                'tau_sigma',
                'in_support',
            ]
            rows = [
                [
                    *r[cols],
                    r[f'{key}_q50'],
                    nuts.loc[r.point, f'{key}_q50'],
                    r.key_z,
                    r.key_width_ratio,
                ]
                for _, r in flagged.iterrows()
            ]
            table = tabulate(
                rows,
                headers=[*cols, 'median', 'median NUTS', 'z', 'width ratio'],
                tablefmt='pipe',
                floatfmt='.3g',
            )
            parts.append(f'## Flagged points\n\n{table}')
        nuts = sub[sub.method == 'NUTS']
        if len(nuts):
            bad = nuts[(nuts.rhat_max > 1.01) | (nuts.divergences > 0)]
            parts.append(
                f'## NUTS diagnostics\n\nmax R-hat {nuts.rhat_max.max():.3f}, min bulk ESS {nuts.ess_bulk_min.min():.0f}; '
                f'points with R-hat > 1.01 or divergences: '
                + (
                    str(bad[['point', 'rhat_max', 'divergences']].to_dict('records'))
                    if len(bad)
                    else 'none'
                )
            )
        parts.append('## Wall-clock\n\n' + '\n'.join(self.wallClock(name, df)))
        path = RESULTS_DIR / f'prior_sensitivity_{name}.md'
        path.write_text('\n\n'.join(parts) + '\n')
        logger.info('saved %s', path)

    def writeAgreementTable(self, frames: dict[str, pd.DataFrame]) -> None:
        cols = ['r', 'sigma_ratio', 'rank_mad', 'delta_loo_nll', 'accept_rate']
        md_rows, tex_rows, reliable = [], [], []
        for name, df in frames.items():
            nuts = df[df.method == 'NUTS']
            n_ok = (
                int(((nuts.rhat_max <= 1.01) & (nuts.divergences == 0)).sum()) if len(nuts) else 0
            )
            reliable.append(
                f'- {name}: NUTS reliable (R-hat <= 1.01, 0 divergences) at {n_ok}/{len(nuts)} fitted points'
            )
            sub = df[df.nuts_subgrid & (df.method != 'NUTS')]
            for method, g in sub.groupby('method', sort=False):
                cells = [medMad(g[c]) if c in g and g[c].notna().any() else 'NA' for c in cols]
                flag = flagCount(g)
                md_rows.append([name, method] + cells + [flag])
                tex_cells = [c.replace(' ± ', r' \pm ') for c in cells]
                tex_rows.append(
                    rf'      \texttt{{{name}}} & \texttt{{{method}}} & '
                    + ' & '.join(f'${c}$' if c != 'NA' else r'\textrm{NA}' for c in tex_cells)
                    + rf' & ${flag}$ \\'
                )
        headers = [
            'dataset',
            'method',
            'r',
            'σ-ratio',
            'rank-MAD',
            'ΔLOO-NLL',
            'IMH acc.',
            'flagged',
        ]
        md = (
            '# E2 agreement with NUTS on the 48-point sub-grid (median ± MAD over prior points)\n\n'
        )
        md += (
            tabulate(md_rows, headers=headers, tablefmt='pipe')
            + '\n\n'
            + '\n'.join(reliable)
            + '\n'
        )
        (RESULTS_DIR / 'prior_sensitivity_agreement.md').write_text(md)
        tex = [
            '% entries: median ± MAD over the 48 NUTS prior points',
            r'\begin{tabular}{ll|cccccc}',
            r'    \toprule',
            r'    dataset & model & $r$ & $\sigma\text{-ratio}$ & $\mathrm{rank\text{-}MAD}$ & '
            r'$\Delta\mathrm{LOO\text{-}NLL}$ & acc. & flagged \\',
            r'    \midrule',
            *tex_rows,
            r'    \bottomrule',
            r'\end{tabular}',
            '',
        ]
        (RESULTS_DIR / 'prior_sensitivity_agreement.tex').write_text('\n'.join(tex))
        logger.info('saved prior_sensitivity_agreement.md/.tex')

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
                    'evidence %s: IS on 288 flow pools of %d, bridge on the NUTS fits', name, S
                )
                continue
            mb = torch.load(self.dataDir(name) / f'mb_{self.cfg.device}.pt', weights_only=False)
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
            path = RESULTS_DIR / f'prior_sensitivity_evidence_{name}.csv'
            df.to_csv(path, index=False)
            logger.info('saved %s', path)

    # --------------------------------------------------------------------------
    # plot

    def plot(self) -> None:
        names = list(self.data)
        if self.cfg.dry_run:
            logger.info(
                'plot: %s from %s',
                Path(self.cfg.fig_dir) / 'prior_sensitivity.pdf',
                [f'prior_sensitivity_{n}.csv' for n in names],
            )
            return
        plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14})
        for stem, sd_prior in FIG_SD_PRIORS.items():
            self._save(self._figure(names, sd_prior), stem)
        if all(
            (RESULTS_DIR / f'prior_sensitivity_evidence_{n}.csv').exists()
            for n in names
            if self.meta[n]['lf'] == 0
        ):
            self.plotEvidence()

    def _figure(self, names: list[str], sd_prior: tuple[str, float]):
        fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names), 7.6), squeeze=False)
        for c, name in enumerate(names):
            df = pd.read_csv(RESULTS_DIR / f'prior_sensitivity_{name}.csv')
            lf = self.meta[name]['lf']
            key = f'b_{DATASETS[name]["key"]}'
            self._panelTauBeta(axes[0, c], df, key, sd_prior, TRAIN_MAX_TAU_FFX[lf])
            self._panelTauSigma(axes[1, c], df, 'sd_Intercept', TRAIN_MAX_TAU_RFX[lf])
            axes[0, c].set_title(DATASETS[name]['title'])
            axes[0, c].set_ylabel(DATASETS[name]['key_label'])
            axes[1, c].set_ylabel(r'$\sigma_{\mathrm{Intercept}}$')
        fam = [('normal', '-', 'o', C_NUTS), ('student', '--', 's', 'white')]
        axes[0, 0].legend(
            [Line2D([], [], color=C_MB, ls=ls) for _, ls, _, _ in fam]
            + [Line2D([], [], color=C_NUTS, marker=mk, mfc=mfc, ls='') for _, _, mk, mfc in fam],
            ['MB, Normal', 'MB, Student-t', 'NUTS, Normal', 'NUTS, Student-t'],
            fontsize=10, loc='best',
        )  # fmt: skip
        sd_ls = {'halfnormal': '-', 'halfstudent': '--', 'exponential': ':'}
        axes[1, 0].legend(
            [Line2D([], [], color=C_MB, ls=ls) for ls in sd_ls.values()]
            + [Line2D([], [], color=C_NUTS, marker=mk, ls='') for mk in SD_MARKERS.values()],
            ['MB, half-Normal', 'MB, half-Student-t', 'MB, Exponential', 'NUTS, HN(2.5)', 'NUTS, Exp(1)'],
            fontsize=10, loc='best',
        )  # fmt: skip
        fig.tight_layout()
        return fig

    def _panelTauBeta(
        self, ax, df: pd.DataFrame, key: str, sd_prior: tuple[str, float], train_max: float
    ) -> None:
        sl = df[(df.sigma_family == sd_prior[0]) & np.isclose(df.tau_sigma, sd_prior[1])]
        for fam, ls, marker, mfc, dx in (
            ('normal', '-', 'o', C_NUTS, 0.96),
            ('student', '--', 's', 'white', 1.04),
        ):
            mb = sl[(sl.method == 'MB') & (sl.ffx_family == fam)].sort_values('tau_beta')
            ax.plot(mb.tau_beta, mb[f'{key}_q50'], color=C_MB, ls=ls, lw=1.6)
            ax.fill_between(
                mb.tau_beta, mb[f'{key}_q05'], mb[f'{key}_q95'], color=C_MB, alpha=0.13, lw=0
            )
            nu = sl[(sl.method == 'NUTS') & (sl.ffx_family == fam)].sort_values('tau_beta')
            yerr = [nu[f'{key}_q50'] - nu[f'{key}_q05'], nu[f'{key}_q95'] - nu[f'{key}_q50']]
            ax.errorbar(
                nu.tau_beta * dx,
                nu[f'{key}_q50'],
                yerr=yerr,
                fmt=marker,
                color=C_NUTS,
                mfc=mfc,
                ms=4.5,
                lw=1,
            )
        self._axis(ax, (TAU_BETA[0] / 1.3, TAU_BETA[-1] * 1.3), train_max, r'$\tau_\beta$')

    def _panelTauSigma(self, ax, df: pd.DataFrame, key: str, train_max: float) -> None:
        sl = df[(df.ffx_family == 'normal') & np.isclose(df.tau_beta, FIG_TAU_BETA)]
        for fam, ls in (('halfnormal', '-'), ('halfstudent', '--'), ('exponential', ':')):
            mb = sl[(sl.method == 'MB') & (sl.sigma_family == fam)].sort_values('tau_sigma')
            ax.plot(mb.tau_sigma, mb[f'{key}_q50'], color=C_MB, ls=ls, lw=1.6)
            ax.fill_between(
                mb.tau_sigma, mb[f'{key}_q05'], mb[f'{key}_q95'], color=C_MB, alpha=0.10, lw=0
            )
        for fam, marker in SD_MARKERS.items():
            nu = sl[(sl.method == 'NUTS') & (sl.sigma_family == fam)]
            yerr = [nu[f'{key}_q50'] - nu[f'{key}_q05'], nu[f'{key}_q95'] - nu[f'{key}_q50']]
            ax.errorbar(
                nu.tau_sigma, nu[f'{key}_q50'], yerr=yerr, fmt=marker, color=C_NUTS, ms=4.5, lw=1
            )
        self._axis(ax, (TAU_SIGMA[0] / 1.4, TAU_SIGMA[-1] * 1.4), train_max, r'$\tau_\sigma$')

    @staticmethod
    def _axis(ax, xlim: tuple[float, float], train_max: float, xlabel: str) -> None:
        ax.set_xscale('log')
        ax.set_xlim(*xlim)
        if train_max < xlim[1]:
            ax.axvspan(train_max, xlim[1], color='0.92', lw=0, zorder=0)  # outside training range
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    def plotEvidence(self) -> None:
        names = [n for n in self.data if self.meta[n]['lf'] == 0]
        fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.0), squeeze=False)
        for ax, name in zip(axes[0], names):
            ev = pd.read_csv(RESULTS_DIR / f'prior_sensitivity_evidence_{name}.csv')
            f, s, ts, tb = BF_REFERENCE
            ref = ev[
                (ev.ffx_family == f)
                & (ev.sigma_family == s)
                & np.isclose(ev.tau_sigma, ts)
                & np.isclose(ev.tau_beta, tb)
            ]
            for (sfam, stau), color in zip(NUTS_SD_PRIORS, (C_MB, PALETTE[0])):
                for fam, ls, marker, mfc in (
                    ('normal', '-', 'o', C_NUTS),
                    ('student', '--', 's', 'white'),
                ):
                    sl = ev[
                        (ev.sigma_family == sfam)
                        & np.isclose(ev.tau_sigma, stau)
                        & (ev.ffx_family == fam)
                    ].sort_values('tau_beta')
                    ax.plot(sl.tau_beta, sl.logev_is - ref.logev_is.item(), color=color, ls=ls, lw=1.6,
                            label=f'MB IS, {fam}, {sfam}({stau:g})')  # fmt: skip
                    ax.plot(
                        sl.tau_beta,
                        sl.logev_bridge - ref.logev_bridge.item(),
                        marker=marker,
                        mfc=mfc,
                        color=C_NUTS,
                        ls='',
                        ms=4.5,
                    )
            ax.axhline(0.0, color='0.5', lw=0.8)
            self._axis(
                ax, (TAU_BETA[0] / 1.3, TAU_BETA[-1] * 1.3), TRAIN_MAX_TAU_FFX[0], r'$\tau_\beta$'
            )
            ax.set_ylabel(r'$\ln \mathrm{BF}$ vs Normal(2.16), HN(2.5)')
            ax.set_title(DATASETS[name]['title'])
            ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, 'prior_sensitivity_bf')

    def _save(self, fig, stem: str) -> None:
        for out in (Path(self.cfg.fig_dir) / f'{stem}.pdf', RESULTS_DIR / f'{stem}.pdf'):
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
