"""Fixed-prior emulation: what does conditioning on the analyst's prior buy?

A fixed-prior neural posterior estimator trained at a reference prior P0 targets p(θ | D, P0),
whatever prior P1 the analyst holds.  metabeta conditioned on P0 targets the same posterior (the
real-data benchmark checks MB(P0) ≈ NUTS(P0) at P0 = Bambi defaults), so MB⁰(P0) scored against
NUTS(P1) emulates the closest fixed-prior amortized method without training one.  The sampled
test sets carry P1: their data were generated and their NUTS fits run under it, and both stay
untouched.  Only the prior handed to the network (and to the IMH target of the P0 check) is
swapped in memory, as in prior_families.py, so no new NUTS fits are needed.

Arms per dataset (``arm`` column):

    flow      raw flow MB⁰ at the condition's prior (identity condition: P1)
    imh_P1    that flow as IMH proposal, target P1 (identity: MB(P1), the paper default).
              The exactness path repairs a fixed-prior network; the acceptance prices it.
    imh_P0    same proposal, target P0: acceptance checks that the flow is accurate *at P0*,
              so a P0 outside the training hyper-prior cannot pass for a fixed-prior failure.
    NUTS      NUTS(P1), the reference (condition '-').

Conditions: P0 as a transform of each dataset's P1 (scale ratio τ₁/τ₀, location, family), or
a fixed P0 shared by all datasets (Bambi defaults; the hyper-prior mode of simulation/prior.py).
Datasets: the first ``--n_datasets`` of the seeded permutation used by
experiments/simulation/likelihood_misspec.selectIndices (sorted, so the cache key is the order).

Output: one row per (dataset, condition, arm) in fixed_prior_{family}_{size}.csv, with
agreement against NUTS(P1) (r, σ-ratio, rank-MAD, ΔLOO-NLL), IMH acceptance, the information
proxy m/q (App. B.7), the realised log10 τ₁/τ₀, and per-level credible-interval hit counts
against the generating parameters, from which EACE is recomputed for any subset.  ``report``
reads only the CSV, so cluster CSVs can be tabulated locally.

Usage (from repo root):
    uv run python experiments/evaluation/fixed_prior.py --family n --sizes small
    uv run python experiments/evaluation/fixed_prior.py --family n --report_only
    uv run python experiments/evaluation/fixed_prior.py --family p --report_only --paper   # table + figure
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from metabeta.evaluation.summary import EvaluationSummary
from metabeta.utils.constants import (
    FFX_FAMILIES,
    FFX_FAMILY_PROBS,
    SIGMA_EPS_FAMILY_PROBS,
    SIGMA_FAMILIES,
    SIGMA_RFX_FAMILY_PROBS,
)
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.device import setDevice
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, REPO_ROOT
from metabeta.utils.logger import setupLogging
from metabeta.utils.posterior_eval import (
    IMH_N_CHAINS,
    fit2proposal,
    loadModel,
    loadOrComputeSummary,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.priors import bambiDefaultPriors
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed

sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM, _fmtMs
from data_poverty import ECE_ALPHAS, globalEntries, localEntries
from likelihood_misspec import FAMILY_NAMES
from real_posterior import _medianMad, computeCorr, computeRankMAD, computeSigmaRatio

logger = logging.getLogger(__name__)

# ==============================================================================
# Globals
# ==============================================================================

DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']

# tag → display label.  Scale tags name τ₁/τ₀: 'wide9' is a P0 nine times wider than P1.
CONDITIONS: dict[str, str] = {
    'wide9': 'τ₁/τ₀=1/9',
    'wide3': 'τ₁/τ₀=1/3',
    'tight3': 'τ₁/τ₀=3',
    'shift2': 'ν₀=ν₁+2τ₁',
    'famrot': 'family+1',
    'bambi': 'Bambi default',
    'hypermode': 'hyper-prior mode',
}
IDENTITY = 'identity'   # P0 = P1: flow = MB⁰(P1), imh_P1 = MB(P1)
SCALE = {'wide9': 9.0, 'wide3': 3.0, 'tight3': 1.0 / 3.0}

# Modes of the skewedBeta hyper-priors in simulation/prior.hypersample, per likelihood family
# (τ_ffx, τ_rfx, τ_ε); ν_ffx's spike-and-slab has its mode at 0.  In the simulator's raw scale:
# Normal data are then standardised by sd_y, and so are these scales (see fixedPrior).
HYPER_MODES: dict[int, tuple[float, float, float | None]] = {
    0: (0.5, 1.0, 2.0),
    1: (0.8, 0.7, None),
    2: (0.5, 0.3, None),
}

ARM_LABELS = {
    'flow': 'MB⁰(P0)',
    'imh_P1': 'MB⁰(P0)+IMH(P1)',
    'imh_P0': 'MB⁰(P0)+IMH(P0)',
    'NUTS': 'NUTS(P1)',
}
AGREE_COLS = ['r', 'sigma_ratio', 'rank_mad', 'delta_nll']
QUARTILES = ['Q1', 'Q2', 'Q3', 'Q4']
FAMILY_LABELS = {'n': 'Gaussian', 'b': 'Bernoulli', 'p': 'Poisson'}

# A condition emulates a fixed-prior network faithfully when the P0 flow is about as good a
# proposal at P0 as the P1 flow is at P1: median IMH(P0) acceptance ≥ this share of the median
# MB(P1) acceptance (same family, same datasets).  Failing rows stay in every table, marked.
VALIDITY_SHARE = 0.5
# valid by the rule but footnoted in the paper table: Bambi's scale 2.5 exceeds the Poisson
# training maximum (App. F.3: τ_β ≤ 1.5, τ_σ ≤ 1.0), so the stand-in is pessimistic
BORDERLINE = {('p', 'bambi')}

FREEZE = """## Frozen design (2026-09-24, before the medium/large/huge runs)

- Primary metrics: σ-ratio of MB⁰(P0) vs NUTS(P1) (full population and lowest m/q quartile),
  rank-MAD, EACE-local, EACE-global where it moves, IMH(P1) acceptance. Null metrics, reported
  in one line: r and ΔLOO-NLL.
- Validity rule: a condition is a valid fixed-prior emulation when the median IMH(P0)
  acceptance is at least half the median MB(P1) acceptance (same family, same datasets,
  pooled over sizes). Failing rows stay in the table, marked, with the acceptance beside them:
  network extrapolation outside the training hyper-prior, not fixed-prior cost.
- No new conditions, no dropped conditions, no re-selection of datasets (first 128 of the
  seed-0 permutation per size). Realised τ₁/τ₀ is reported as is, not equalised across
  families.

## Checks

- IMH acceptance is read off the saved chains (a step accepts iff the global state moves).
  Against MetropolisSampler's own accept_rate on the same run (Gaussian small, 32 datasets,
  identity / Bambi / τ₁/τ₀=3): max |difference| 0.003, correlation 1.000.
- Hyper-prior-mode scaling: stored τ × sd_y in the small test sets reproduces the App. F.3
  hyper-prior quantiles (5–95%) within 3% for all three families (outcome calibration shrinks
  only the upper tail), so the hyper-prior-mode P0 is the App. F.3 mode divided by sd_y
  (Gaussian; sd_y = 1 for Bernoulli and Poisson).
"""


# ==============================================================================
# Study
# ==============================================================================


class FixedPriorStudy:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.family = cfg.family
        self.lf = LF_FROM_FAM[cfg.family]
        self.method = posthocDefaults(self.lf)[0]   # imhMarginal (Normal) / imhPM (GLMM)
        self.device = setDevice(cfg.device)
        self.outdir = Path(cfg.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Priors

    def fixedPrior(self, batch: dict[str, torch.Tensor], tag: str) -> dict[str, torch.Tensor]:
        """Condition's P0 fields for every dataset (padded entries stay zero).

        The LKJ concentration eta_rfx is kept from P1: it switches the correlated model on or
        off, a modelling choice rather than a prior scale, and NUTS(P1) fitted that structure.
        """
        B = batch['X'].shape[0]
        mask_d = batch['mask_d'].float()   # (B, d)
        mask_q = batch['mask_q'].float()   # (B, q)
        d, q = mask_d.shape[-1], mask_q.shape[-1]
        if tag == 'bambi':
            ref = bambiDefaultPriors(d, q, self.lf)
            tau_ffx = torch.as_tensor(ref['tau_ffx'], dtype=torch.float32)
            tau_rfx = torch.as_tensor(ref['tau_rfx'], dtype=torch.float32)
            tau_eps = float(ref['tau_eps']) if 'tau_eps' in ref else None
            fams = (int(ref['family_ffx']), int(ref['family_sigma_rfx']))
            fam_eps = int(ref['family_sigma_eps']) if 'family_sigma_eps' in ref else None
            scale = torch.ones(B)
        elif tag == 'hypermode':
            t_ffx, t_rfx, tau_eps = HYPER_MODES[self.lf]
            tau_ffx, tau_rfx = torch.full((d,), t_ffx), torch.full((q,), t_rfx)
            fams = (int(np.argmax(FFX_FAMILY_PROBS)), int(np.argmax(SIGMA_RFX_FAMILY_PROBS)))
            fam_eps = int(np.argmax(SIGMA_EPS_FAMILY_PROBS))
            scale = 1.0 / batch['sd_y'].float()   # (B,) — the simulator's standardisation
        else:
            raise KeyError(f'no fixed prior for tag {tag!r}')
        out = {
            'nu_ffx': torch.zeros_like(batch['nu_ffx']),
            'tau_ffx': tau_ffx * scale[:, None] * mask_d,
            'tau_rfx': tau_rfx * scale[:, None] * mask_q,
            'family_ffx': torch.full_like(batch['family_ffx'], fams[0]),
            'family_sigma_rfx': torch.full_like(batch['family_sigma_rfx'], fams[1]),
        }
        if 'tau_eps' in batch:
            out['tau_eps'] = tau_eps * scale
            out['family_sigma_eps'] = torch.full_like(batch['family_sigma_eps'], fam_eps)
        return out

    def conditionPrior(self, batch: dict[str, torch.Tensor], tag: str) -> dict[str, torch.Tensor]:
        """Shallow copy of ``batch`` carrying P0 instead of P1.

        ``stats`` is dropped: the analytical MAP/EB context is a function of the prior and must
        be recomputed under P0 (as in prior_families.setPriorFamily).
        """
        out = dict(batch)
        out.pop('stats', None)
        if tag in SCALE:
            for key in ('tau_ffx', 'tau_rfx', 'tau_eps'):
                if key in out:
                    out[key] = batch[key] * SCALE[tag]
        elif tag == 'shift2':
            out['nu_ffx'] = batch['nu_ffx'] + 2.0 * batch['tau_ffx']
        elif tag == 'famrot':
            for key, n in (
                ('family_ffx', len(FFX_FAMILIES)),
                ('family_sigma_rfx', len(SIGMA_FAMILIES)),
                ('family_sigma_eps', len(SIGMA_FAMILIES)),
            ):
                if key in out:
                    out[key] = (batch[key] + 1) % n
        else:
            out.update(self.fixedPrior(batch, tag))
        return out

    @staticmethod
    def logTauRatio(p1: dict[str, torch.Tensor], p0: dict[str, torch.Tensor]) -> np.ndarray:
        """Per dataset, median log10 τ₁/τ₀ over the active ffx and rfx scales."""
        active = torch.cat([p1['mask_d'], p1['mask_q']], -1).bool()   # (B, d+q)
        t1 = torch.cat([p1['tau_ffx'], p1['tau_rfx']], -1)
        t0 = torch.cat([p0['tau_ffx'], p0['tau_rfx']], -1)
        log_ratio = torch.log10(t1 / t0).where(active, torch.nan)
        return log_ratio.nanmedian(-1).values.numpy()

    # --------------------------------------------------------------------------
    # Loading

    def loadBatch(self, size: str) -> tuple[dict[str, torch.Tensor], np.ndarray, Path]:
        """Selected datasets of {size}-{family}-sampled with their NUTS fits and P1 stats."""
        data_path = DATA_DIR / f'{size}-{self.family}-sampled' / 'test.fit.npz'
        col = Collection(data_path, permute=False, max_d=self.max_d, max_q=self.max_q)
        B = len(col)
        if self.cfg.n_datasets > B:
            raise ValueError(f'n_datasets={self.cfg.n_datasets} exceeds test-set size {B}')
        # same selection as experiments/simulation/likelihood_misspec.selectIndices
        idx = np.sort(np.random.default_rng(self.cfg.seed).permutation(B)[: self.cfg.n_datasets])
        mask = np.zeros(B, dtype=bool)
        mask[idx] = True
        batch = collateGrouped([col[i] for i in idx])
        # P1 analytical stats precomputed in the sibling test.npz (the paper-default context)
        base = Collection(
            data_path.with_name('test.npz'), permute=False, max_d=self.max_d, max_q=self.max_q
        )
        batch['stats'] = collateGrouped([base[i] for i in idx])['stats']
        return batch, mask, data_path

    def nutsLoo(self, proposal, batch, data_path: Path, mask: np.ndarray) -> np.ndarray:
        """NUTS(P1) LOO-NLL: sliced from a full-set cache when present (minutes to recompute).

        Both names hold the same rescaled full-split summary (identical to 1e-6 on small-n).
        """
        for name in (f'summary_test_nuts_lf{self.lf}_rs1_all.pt', 'summary_test_nuts.pt'):
            full = data_path.parent / name
            if full.exists():
                loo = EvaluationSummary.load(full).per_dataset.loo_nll.float().numpy()
                if loo.shape[0] == mask.shape[0]:
                    return loo[mask]
        summary = loadOrComputeSummary(
            proposal,
            batch,
            data_path,
            'nuts',
            mask,
            self.lf,
            True,
            summary_chunk_size=self.cfg.summary_chunk_size,
        )
        return summary.per_dataset.loo_nll.float().numpy()

    # --------------------------------------------------------------------------
    # Scoring

    @staticmethod
    def acceptRate(proposal: Proposal) -> np.ndarray:
        """Per-dataset IMH acceptance, read off the chains: a step accepts iff the state moves."""
        b, s, D = proposal.samples_g.shape
        chains = proposal.samples_g.reshape(b, IMH_N_CHAINS, s // IMH_N_CHAINS, D)
        moved = (chains[:, :, 1:] != chains[:, :, :-1]).any(-1)   # (b, C, T-1)
        return moved.float().mean((1, 2)).numpy()

    def hitCounts(
        self, proposal: Proposal, batch: dict[str, torch.Tensor]
    ) -> dict[str, np.ndarray]:
        """Per dataset: active entries and credible-interval hits per level (global / local)."""
        B = batch['X'].shape[0]
        ones = np.ones(B, dtype=bool)
        g_masks = [batch['mask_d'].bool(), batch['mask_q'].bool()]
        if self.lf == 0:
            g_masks.append(torch.ones(B, 1, dtype=torch.bool))
        g_ids = np.concatenate(
            [np.broadcast_to(np.arange(B)[:, None], m.shape)[m.numpy()] for m in g_masks]
        )
        mask_mq = batch['mask_mq'].bool().numpy()
        l_ids = np.broadcast_to(np.arange(B)[:, None, None], mask_mq.shape)[mask_mq]
        out = {}
        for kind, ids, e in (
            ('g', g_ids, globalEntries(proposal, batch, ones, self.lf)),
            ('l', l_ids, localEntries(proposal, batch, ones)),
        ):
            out[f'{kind}_n'] = np.bincount(ids, minlength=B)
            for j, a in enumerate(ECE_ALPHAS):
                out[f'{kind}_in{a:g}'] = np.bincount(ids, weights=e['inside'][:, j], minlength=B)
        return out

    def score(
        self,
        proposal: Proposal,
        method: str,
        variant: str,
        ctx: dict,
    ) -> dict[str, np.ndarray]:
        """Per-dataset columns for one arm (proposal and batch in the rescaled data space)."""
        batch = ctx['batch']
        summary = loadOrComputeSummary(
            proposal,
            batch,
            ctx['data_path'],
            method,
            ctx['mask'],
            self.lf,
            True,
            ckpt_dir=ctx['ckpt_dir'],
            prefix=self.cfg.prefix,
            n_samples=self.cfg.n_samples,
            seed=self.cfg.seed,
            summary_chunk_size=self.cfg.summary_chunk_size,
            variant=variant,
        )
        loo = summary.per_dataset.loo_nll.float().numpy()
        cols = {
            'r': computeCorr(proposal, ctx['nuts'], batch),
            'sigma_ratio': computeSigmaRatio(proposal, ctx['nuts'], batch),
            'rank_mad': computeRankMAD(proposal, ctx['nuts'], batch),
            'loo': loo,
            'delta_nll': loo - ctx['nuts_loo'],
            'accept': self.acceptRate(proposal) if method != 'mb' else np.full(len(loo), np.nan),
        }
        return cols | self.hitCounts(proposal, batch)

    # --------------------------------------------------------------------------
    # Collection

    def csvPath(self, family: str, size: str) -> Path:
        return self.outdir / f'fixed_prior_{family}_{size}.csv'

    def collectSize(self, size: str) -> None:
        """Evaluate one size; its CSV is rewritten after every condition, so a job that runs
        out of time still leaves a table of the conditions it finished."""
        seed = BEST_SEEDS.get((FAMILY_NAMES[self.family], size))
        if seed is None:
            logger.warning('%s-%s: no BEST_SEEDS checkpoint — skipping', size, self.family)
            return
        ckpt_dir = _ckpt_dir(FAMILY_NAMES[self.family], size, seed)
        model, model_cfg = loadModel(ckpt_dir, self.cfg.prefix, self.device)
        self.max_d, self.max_q = model_cfg.max_d, model_cfg.max_q
        batch, mask, data_path = self.loadBatch(size)
        B = batch['X'].shape[0]
        conv = nutsConvergeMask(batch, mode=self.cfg.convergence_mode).astype(bool)
        logger.info('%s-%s: %d datasets, %d NUTS-converged', size, self.family, B, conv.sum())

        nuts = fit2proposal(batch, 'nuts')
        nuts.rescale(batch['sd_y'])
        batch_rs = rescaleData(batch)
        ctx = {
            'batch': batch_rs,
            'data_path': data_path,
            'mask': mask,
            'ckpt_dir': ckpt_dir,
            'nuts': nuts,
            'nuts_loo': self.nutsLoo(nuts, batch_rs, data_path, mask),
        }
        q_act = batch['mask_q'].bool().sum(-1).clamp(min=1).float()
        base = {
            'size': size,
            'dataset': np.flatnonzero(mask),
            'm': batch['m'].numpy(),
            'n': batch['n'].numpy(),
            'mq': (batch['m'].float() / q_act).numpy(),
            'conv': conv,
        }
        frames = [
            pd.DataFrame(
                base
                | {'condition': '-', 'arm': 'NUTS', 'log_tau_ratio': 0.0}
                | self.hitCounts(nuts, batch_rs)
            )
        ]

        def run(tag: str, batch_p0: dict[str, torch.Tensor]) -> None:
            variant = '' if tag == IDENTITY else f'fp-{tag}'
            # raw flow at the condition's prior, rescaled to the data space (cached per variant)
            flow, _ = loadOrSampleMB(
                model,
                batch_p0,
                data_path,
                ckpt_dir,
                self.cfg.prefix,
                self.cfg.n_samples,
                self.cfg.batch_size,
                self.cfg.seed,
                self.device,
                mask,
                variant=variant,
            )
            flow.rescale(batch_p0['sd_y'])
            arms = {'flow': (flow, 'mb', variant)}
            targets = {'imh_P1': (batch_rs, variant)}
            if tag != IDENTITY:
                targets['imh_P0'] = (rescaleData(batch_p0), f'{variant}-atP0')
            for arm, (target, v) in targets.items():
                imh, _ = loadOrRefine(
                    self.method,
                    flow,
                    target,
                    data_path,
                    ckpt_dir,
                    self.cfg.prefix,
                    self.cfg.n_samples,
                    self.cfg.seed,
                    self.lf,
                    True,
                    mask,
                    self.cfg.batch_size,
                    variant=v,
                    device=self.device,
                )
                arms[arm] = (imh, self.method, v)
            ratio = self.logTauRatio(batch, batch_p0)
            for arm, (proposal, method, v) in arms.items():
                logger.info('%s-%s: scoring %s / %s', size, self.family, tag, arm)
                cols = self.score(proposal, method, v, ctx)
                frames.append(
                    pd.DataFrame(
                        base | {'condition': tag, 'arm': arm, 'log_tau_ratio': ratio} | cols
                    )
                )

        path = self.csvPath(self.family, size)
        for tag in [IDENTITY] + self.cfg.conditions:
            run(tag, batch if tag == IDENTITY else self.conditionPrior(batch, tag))
            df = pd.concat(frames, ignore_index=True)
            df.insert(0, 'family', self.family)
            df.to_csv(path, index=False)
            logger.info('%s: %d rows after %s', path.name, len(df), tag)

    def go(self) -> None:
        for size in self.cfg.sizes:
            self.collectSize(size)

    # --------------------------------------------------------------------------
    # Report

    def load(self, family: str) -> pd.DataFrame:
        """All size CSVs of a family, NUTS-converged datasets only, with m/q quartiles.

        Quartile edges come from one row per dataset (its NUTS row), so every dataset weighs
        the same whichever conditions a partial run finished.
        """
        paths = [self.csvPath(family, s) for s in DEFAULT_SIZES]
        paths = [p for p in paths if p.exists()]
        if not paths:
            raise FileNotFoundError(f'no fixed_prior_{family}_<size>.csv in {self.outdir}')
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        df = df[df['conv']]   # paired: every arm on the NUTS-converged datasets
        _, edges = pd.qcut(df.loc[df['arm'] == 'NUTS', 'mq'], 4, retbins=True)
        return df.assign(mq_bin=pd.cut(df['mq'], edges, labels=QUARTILES, include_lowest=True))

    @staticmethod
    def eace(df: pd.DataFrame, kind: str) -> float:
        n = df[f'{kind}_n'].sum()
        if n == 0:
            return float('nan')
        cov = np.array([df[f'{kind}_in{a:g}'].sum() / n for a in ECE_ALPHAS])
        return float(np.mean(np.abs(cov - (1.0 - np.array(ECE_ALPHAS)))))

    @staticmethod
    def medMad(x: pd.Series) -> str:
        return _fmtMs(_medianMad(x.to_numpy(dtype=float)), 3)

    @staticmethod
    def label(tag: str) -> str:
        return 'P1 (identity)' if tag == IDENTITY else CONDITIONS.get(tag, tag)

    def conditionStats(self, df: pd.DataFrame, family: str) -> list[dict]:
        """Headline numbers per condition: MB⁰ at the condition's prior against NUTS(P1), the
        IMH acceptance towards P1 and towards P0, and the validity rule."""

        def arm(tag: str, name: str) -> pd.DataFrame:
            return df[(df['condition'] == tag) & (df['arm'] == name)]

        p1_accept = arm(IDENTITY, 'imh_P1')['accept'].median()
        stats = []
        for tag in [IDENTITY] + [t for t in CONDITIONS if t in set(df['condition'])]:
            flow = arm(tag, 'flow')
            accept_p0 = np.nan if tag == IDENTITY else arm(tag, 'imh_P0')['accept'].median()
            stats.append(
                {
                    'tag': tag,
                    'tau_ratio': 10 ** flow['log_tau_ratio'].median(),
                    'sigma_ratio': flow['sigma_ratio'].median(),
                    'sigma_ratio_q1': flow.loc[flow['mq_bin'] == 'Q1', 'sigma_ratio'].median(),
                    'rank_mad': flow['rank_mad'].median(),
                    'eace_l': self.eace(flow, 'l'),
                    'eace_g': self.eace(flow, 'g'),
                    'r': flow['r'].median(),
                    'delta_nll': flow['delta_nll'].median(),
                    'accept_p1': arm(tag, 'imh_P1')['accept'].median(),
                    'accept_p0': accept_p0,
                    'share': accept_p0 / p1_accept,
                    'valid': tag == IDENTITY or accept_p0 >= VALIDITY_SHARE * p1_accept,
                    'borderline': (family, tag) in BORDERLINE,
                }
            )
        return stats

    def summaryRows(self, df: pd.DataFrame) -> list[list]:
        rows = []
        for (tag, arm), g in df.groupby(['condition', 'arm'], sort=False):
            arm_label = (
                {'flow': 'MB⁰(P1)', 'imh_P1': 'MB(P1)'}[arm] if tag == IDENTITY else ARM_LABELS[arm]
            )
            rows.append(
                [self.label(tag), arm_label, len(g)]
                + [self.medMad(g[c]) for c in AGREE_COLS + ['accept']]
                + [f'{self.eace(g, "g"):.3f}', f'{self.eace(g, "l"):.3f}']
            )
        return rows

    def report(self) -> None:
        df = self.load(self.family)
        stats = self.conditionStats(df, self.family)
        headers = ['P0', 'arm', 'n', 'r ↑', 'σ-ratio →1', 'rank-MAD ↓', 'ΔLOO-NLL ↓', 'accept']
        headers += ['EACE-g ↓', 'EACE-l ↓']
        validity = [
            [
                self.label(s['tag']),
                f'{s["accept_p0"]:.3f}',
                f'{s["share"]:.2f}',
                'yes' if s['valid'] else 'NO: network extrapolation, not fixed-prior cost',
            ]
            for s in stats
            if s['tag'] != IDENTITY
        ]
        md = [
            f'# Fixed-prior emulation ({FAMILY_LABELS[self.family]})\n',
            FREEZE,
            '## Results\n',
            f'Sizes: {", ".join(df["size"].unique())}. Reference NUTS(P1) '
            f'({self.cfg.convergence_mode}), converged subset; median ± MAD per dataset, EACE '
            'pooled over entries vs the generating parameters.\n',
            tabulate(self.summaryRows(df), headers=headers, tablefmt='pipe', stralign='right'),
            '',
            f'## Emulation validity (IMH(P0) acceptance vs MB(P1) acceptance '
            f'{stats[0]["accept_p1"]:.3f}; valid at share ≥ {VALIDITY_SHARE})\n',
            tabulate(
                validity,
                headers=['P0', 'IMH(P0) accept', 'share', 'valid'],
                tablefmt='pipe',
                stralign='right',
            ),
            '',
        ]
        # dose-response over information per parameter: quartiles of m/q (App. B.7)
        for tag in [IDENTITY] + [t for t in CONDITIONS if t in set(df['condition'])]:
            sub = df[(df['condition'] == tag) | (df['arm'] == 'NUTS')]
            rows = []
            for (arm, qb), g in sub.groupby(['arm', 'mq_bin'], sort=False, observed=True):
                rows.append(
                    [
                        arm,
                        qb,
                        len(g),
                        self.medMad(g['sigma_ratio']),
                        self.medMad(g['delta_nll']),
                        self.medMad(g['accept']),
                        f'{self.eace(g, "g"):.3f}',
                        f'{self.eace(g, "l"):.3f}',
                    ]
                )
            rows.sort(key=lambda r: (r[0], r[1]))
            md += [
                f'## {self.label(tag)} by m/q quartile\n',
                tabulate(
                    rows,
                    headers=[
                        'arm',
                        'm/q',
                        'n',
                        'σ-ratio',
                        'ΔLOO-NLL',
                        'accept',
                        'EACE-g',
                        'EACE-l',
                    ],
                    tablefmt='pipe',
                    stralign='right',
                ),
                '',
            ]
        text = '\n'.join(md)
        print(text)
        (self.outdir / f'fixed_prior_{self.family}.md').write_text(text + '\n')

    # --------------------------------------------------------------------------
    # Paper artefacts (all families with results; larger sizes drop in as their CSVs land)

    TEX_LABELS = {
        IDENTITY: r'$P_0 = P_1$',
        'wide9': r'$\tau_1/\tau_0 = 1/9$',
        'wide3': r'$\tau_1/\tau_0 = 1/3$',
        'tight3': r'$\tau_1/\tau_0 = 3$',
        'shift2': r'$\nu_0 = \nu_1 + 2\tau_1$',
        'famrot': r'family$+1$',
        'bambi': r'Bambi default',
        'hypermode': r'hyper-prior mode',
    }

    def paper(self) -> None:
        families = [
            f for f in FAMILY_LABELS if any(self.csvPath(f, s).exists() for s in DEFAULT_SIZES)
        ]
        frames = {f: self.load(f) for f in families}
        stats = {f: self.conditionStats(frames[f], f) for f in families}
        (self.outdir / 'fixed_prior.tex').write_text(self.renderTex(frames, stats))
        self.plot(frames, stats, self.outdir / 'fixed_prior.pdf')
        logger.info('Saved fixed_prior.tex and fixed_prior.pdf to %s', self.outdir)

    def renderTex(self, frames: dict[str, pd.DataFrame], stats: dict[str, list[dict]]) -> str:
        """Families × conditions; rows failing the validity rule are greyed and daggered."""

        def ratio(r: float) -> str:
            if r < 0.95:
                return f'1/{1 / r:.0f}' if r < 0.51 else f'1/{1 / r:.1f}'
            return f'{r:.0f}' if abs(r - round(r)) < 0.05 else f'{r:.1f}'

        lines = [
            r'\begin{tabular}{ll r ccccc cc}',
            r'    \toprule',
            r'     & & & \multicolumn{5}{c}{\MBz{}($P_0$) vs.\ \texttt{NUTS}($P_1$)}'
            r' & \multicolumn{2}{c}{IMH acceptance} \\',
            r'    \cmidrule(lr){4-8}\cmidrule(lr){9-10}',
            r'    $\mathrm{family}$ & $P_0$ & $\tau_1/\tau_0$ & $\sigma\text{-ratio}$'
            r' & $\sigma\text{-ratio}_{Q1}$ & $\mathrm{rank\text{-}MAD}$ & $\mathrm{EACE}_l$'
            r' & $\mathrm{EACE}_g$ & target $P_1$ & target $P_0$ \\',
        ]
        for family, rows in stats.items():
            lines.append(r'    \midrule')
            nuts = frames[family][frames[family]['arm'] == 'NUTS']
            for j, s in enumerate(rows):
                mark = '' if s['valid'] else r'$^\dagger$'
                mark += r'$^\ddagger$' if s['borderline'] else ''
                cells = [
                    self.TEX_LABELS[s['tag']] + mark,
                    ratio(s['tau_ratio']),
                    f'${s["sigma_ratio"]:.3f}$',
                    f'${s["sigma_ratio_q1"]:.3f}$',
                    f'${s["rank_mad"]:.3f}$',
                    f'${s["eace_l"]:.3f}$',
                    f'${s["eace_g"]:.3f}$',
                    f'${s["accept_p1"]:.2f}$',
                    '---' if s['tag'] == IDENTITY else f'${s["accept_p0"]:.2f}$',
                ]
                if not s['valid']:
                    cells = [rf'\textcolor{{gray}}{{{c}}}' for c in cells]
                lead = FAMILY_LABELS[family] if j == 0 else ''
                lines.append(f'    {lead} & ' + ' & '.join(cells) + r' \\')
            ref = [r'\texttt{NUTS}($P_1$)', '---', '---', '---', '---']
            ref += [f'${self.eace(nuts, "l"):.3f}$', f'${self.eace(nuts, "g"):.3f}$', '---', '---']
            lines.append('     & ' + ' & '.join(ref) + r' \\')
        lines += [r'    \bottomrule', r'\end{tabular}', '']
        return '\n'.join(lines)

    def plot(
        self, frames: dict[str, pd.DataFrame], stats: dict[str, list[dict]], path: Path
    ) -> None:
        """(a) MB⁰(Bambi) σ-ratio across m/q quartiles; (b) global EACE at P1, Bambi, τ₁/τ₀=3."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # dataviz reference palette, categorical slots 1-3 (validated all-pairs, light mode)
        colors = {'n': '#2a78d6', 'b': '#eb6834', 'p': '#1baf7a'}
        markers = {'n': 'o', 'b': 's', 'p': '^'}
        ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
        plt.rcParams.update(
            {'font.size': 8, 'pdf.fonttype': 42, 'axes.edgecolor': muted, 'axes.labelcolor': ink}
        )
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5.5, 2.15), constrained_layout=True)

        x = np.arange(len(QUARTILES))
        ax_l.axhline(1.0, color=muted, lw=0.8, ls='--', zorder=1)
        for f, df in frames.items():
            flow = df[(df['condition'] == 'bambi') & (df['arm'] == 'flow')]
            y = flow.groupby('mq_bin', observed=True)['sigma_ratio'].median().reindex(QUARTILES)
            ax_l.plot(
                x,
                y.values,
                color=colors[f],
                lw=2,
                marker=markers[f],
                ms=5,
                zorder=3,
                label=FAMILY_LABELS[f],
                markeredgecolor='white',
                markeredgewidth=0.8,
            )
        ax_l.set_xticks(x, QUARTILES)
        ax_l.set_xlabel('$m/q$ quartile (low → high information)')
        ax_l.set_ylabel(r'$\sigma$-ratio of MB$^0$($P_0$) vs. NUTS($P_1$)')
        ax_l.set_title('a   Bambi-default $P_0$: posterior width', loc='left', fontsize=8)
        ax_l.legend(frameon=False, loc='upper right', handlelength=1.8)

        tags = [IDENTITY, 'bambi', 'tight3']
        ticks = ['$P_0=P_1$', 'Bambi', r'$\tau_1/\tau_0=3$']
        offsets = dict(zip(frames, np.linspace(-0.22, 0.22, len(frames))))
        for f in frames:
            by_tag = {s['tag']: s for s in stats[f]}
            xs = np.arange(len(tags)) + offsets[f]
            ys = [by_tag[t]['eace_g'] for t in tags]
            ax_r.vlines(xs, 0, ys, color=colors[f], lw=2, zorder=2)
            ax_r.scatter(
                xs,
                ys,
                color=colors[f],
                marker=markers[f],
                s=28,
                zorder=3,
                edgecolors='white',
                linewidths=0.8,
                label=FAMILY_LABELS[f],
            )
        ax_r.set_xticks(np.arange(len(tags)), ticks)
        ax_r.set_ylim(bottom=0)
        ax_r.set_ylabel(r'global EACE of MB$^0$($P_0$)')
        ax_r.set_title('b   Global calibration error', loc='left', fontsize=8)

        for ax in (ax_l, ax_r):
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color=grid, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(colors=muted, labelcolor=ink, length=3)
        fig.savefig(path)
        plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description='Fixed-prior emulation (MB⁰ at P0 vs NUTS at P1).')
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--conditions', type=str, nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument('--n_datasets', type=int, default=128)
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--convergence_mode', type=str, default='strict', choices=['liberal', 'strict'])
    parser.add_argument('--report_only', action='store_true', help='tabulate the existing CSVs')
    parser.add_argument('--paper', action='store_true', help='also write the cross-family table and figure')
    parser.add_argument('--outdir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--verbosity', type=int, default=1)
    # fmt: on
    return parser.parse_args()


if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    study = FixedPriorStudy(cfg)
    if not cfg.report_only:
        study.go()
    study.report()
    if cfg.paper:
        study.paper()
