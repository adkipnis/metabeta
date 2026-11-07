"""
experiments/evaluation/power_analysis.py - Simulation-based power analysis of a
cluster-randomized trial with MetaBeta (MB), checked against NUTS.

Schools are randomized to a program, half of them get it, and each of 20 students per school
yields one outcome. A trial detects the program effect when the 95% credible interval of
`program` excludes 0; power is the share of simulated trials that detect it (CONTEXT.md).

Designs:
  gaussian  the demo withdrawn from 0.6.1: test score, effect 0.5 SD, intraclass correlation
            0.2, the library's default priors; its exact power is that of the two-sample t-test
            on school means (noncentral t, df m - 2)
  binary    pass or fail, control pass rate 50%, odds ratio 1.8 (power) or 1 (type I error),
            SD of the school log-odds 0.5, under a weakly informative and a skeptical prior

Every analysis prior lies inside the checkpoint's training prior (``checkTrainingRange``).

Stages:
  mb        simulate every trial and fit all of them in one batched ``Api.sample`` call; per
            trial and prior, record the interval of `program`, the detection and the safeguards
            (mb.csv) and the call's wall time (timing.json); export the first ``--n_nuts``
            replicates of every cell as a fit.py batch (outputs/data/power-{name}/test.npz)
            together with MB's draws of it
  compare   after fit.py ran the NUTS ladder on that batch (scripts/power-nuts.sh): MB vs NUTS
            with the metrics of real_posterior.py (nuts2 reference on the datasets where it
            converged, the composite nuts for time) and the detections of both (compare.csv)
  report    power per cell with its Monte Carlo standard error, the exact power (gaussian) and
            the pass criterion fixed before the run (report.md)
  figure    detection rate vs schools per prior, MB and NUTS (power_{name}.pdf)

Usage (from repo root):
    uv run python experiments/evaluation/power_analysis.py --design gaussian --stage mb --device cuda
    uv run python experiments/evaluation/power_analysis.py --design binary --stage report
"""

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D
from scipy import stats

from metabeta import Api
from metabeta.evaluation.summary import getSummary
from metabeta.utils.dataloader import Collection, collateGrouped, subsetBatch
from metabeta.utils.evaluation import nutsConverged, subsetProposal
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR
from metabeta.utils.fits import REFERENCE_TAG
from metabeta.utils.padding import aggregate
from metabeta.utils.plot import PALETTE, VECTOR_RASTER_DPI
from metabeta.utils.posterior_eval import fit2proposal
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.results import Proposal

# sibling experiment script (this directory is sys.path[0] at run time)
from real_posterior import _medianMad, computeCorr, computeRankMAD, computeSigmaRatio

# =============================================================================
# Designs

N_STUDENTS = 20
SEED = 2026_11_07
TIMING_TAG = 'nuts'  # NUTS time: the escalating composite, as in real_posterior.py

# the paper figures' colors and layout (prior_sensitivity.py): full text width at a 9 in canvas
C_MB, C_NUTS = PALETTE[4], PALETTE[3]
FIG_RC = {'font.size': 11.5, 'axes.titlesize': 11.5, 'axes.labelsize': 11.5, 'legend.fontsize': 10}
FIG_W, PANEL_H, LEGEND_H = 9.0, 2.8, 0.2

# log-odds scale: bambi's Bernoulli defaults, with the school SD moved inside the training range;
# the skeptical prior on `program` puts 5% on OR >= 1.8 (Spiegelhalter et al., 1994)
WEAK = {
    'fixed': {'Intercept': {'nu': 0.0, 'tau': 1.5}, 'program': {'nu': 0.0, 'tau': 1.0}},
    'random_sd': {'Intercept': {'tau': 1.0}},
}
SKEPTICAL = {
    'fixed': {'Intercept': {'nu': 0.0, 'tau': 1.5}, 'program': {'nu': 0.0, 'tau': 0.36}},
    'random_sd': {'Intercept': {'tau': 1.0}},
}

DESIGNS = {
    'gaussian': dict(
        family='gaussian',
        formula='score ~ program + (1 | school)',
        m=(10, 20, 40),
        effects=(0.5,),  # program - control, in SDs of the score
        priors={'default': None},
        icc=0.2,
    ),
    'binary': dict(
        family='bernoulli',
        formula='passed ~ program + (1 | school)',
        m=(10, 20, 40, 60, 80),
        effects=(1.8, 1.0),  # odds ratios
        priors={'weak': WEAK, 'skeptical': SKEPTICAL},
        # Bradley's bound needs a prior that does not favor 0; the skeptical prior's type I error
        # is checked against NUTS like power (Alex, 2026-11-07, after the pilot)
        bradley=('weak',),
        p_control=0.5,
        sd_school=0.5,
    ),
}

# Training prior of the released checkpoints: the scale ranges of hypersample in
# metabeta/simulation/prior.py:34-51, by likelihood family. Location and family hyperparameters
# have full support there (spike-and-slab locations, every family drawn with positive weight).
TRAINED = {
    0: {'tau_ffx': (0.01, 4.0), 'tau_rfx': (0.01, 5.0), 'tau_eps': (0.05, 8.0)},
    1: {'tau_ffx': (0.01, 3.0), 'tau_rfx': (0.01, 2.0)},
    2: {'tau_ffx': (0.01, 1.5), 'tau_rfx': (0.01, 1.0)},
}


def checkTrainingRange(ds: dict[str, np.ndarray]) -> None:
    """Fail loudly when a prepared dataset's prior scales leave the training prior."""
    for key, (lo, hi) in TRAINED[int(ds['likelihood_family'])].items():
        value = np.atleast_1d(ds[key])
        if not ((lo <= value) & (value <= hi)).all():
            raise ValueError(f'{key} = {value} lies outside the training range [{lo}, {hi}]')


def exactPower(effect: float, icc: float, m: np.ndarray) -> np.ndarray:
    """Power of the two-sided 5% t-test on school means, m / 2 schools per arm (noncentral t)."""
    se = np.sqrt(4 * (icc + (1 - icc) / N_STUDENTS) / m)  # SE of program - control means
    df = m - 2
    crit = stats.t.ppf(0.975, df)
    return stats.nct.sf(crit, df, effect / se) + stats.nct.cdf(-crit, df, effect / se)


def detects(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (lo > 0) | (hi < 0)


# =============================================================================
class PowerAnalysis:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.design = DESIGNS[cfg.design]
        self.m = tuple(cfg.m or self.design['m'])
        self.effects = tuple(cfg.effects or self.design['effects'])
        self.name = cfg.name or cfg.design
        self.outdir = Path(cfg.outdir) / self.name
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.batch_path = DATA_DIR / f'power-{self.name}' / 'test.npz'
        self.lf = {'gaussian': 0, 'bernoulli': 1}[self.design['family']]

    # -------------------------------------------------------------------------
    # Simulation

    def simulate(self, m: int, effect: float, rep: int) -> pd.DataFrame:
        """One trial with m schools, half of them in the program; its own seed per cell and
        replicate, so the NUTS batch rebuilds no state from the MB run."""
        cell = [SEED, list(DESIGNS).index(self.cfg.design), m, round(100 * effect), rep]
        rng = np.random.default_rng(cell)
        school = np.repeat(np.arange(m), N_STUDENTS)  # (n,) each student's school
        program = rng.permutation(np.repeat([0, 1], m // 2))  # (m,) 1 for program schools
        if self.lf == 0:
            icc = self.design['icc']
            u = rng.normal(0, np.sqrt(icc), size=m)  # (m,) school deviations
            eps = rng.normal(0, np.sqrt(1 - icc), size=school.size)  # (n,)
            y = effect * program[school] + u[school] + eps
            outcome = 'score'
        else:
            u = rng.normal(0, self.design['sd_school'], size=m)  # (m,) school log-odds
            p0 = self.design['p_control']
            eta = np.log(p0 / (1 - p0)) + np.log(effect) * program[school] + u[school]
            y = (rng.random(school.size) < 1 / (1 + np.exp(-eta))).astype(int)
            outcome = 'passed'
        return pd.DataFrame(
            {outcome: y, 'program': program[school], 'school': [f's{g:02d}' for g in school]}
        )

    def trials(self) -> dict[tuple[int, float, int], pd.DataFrame]:
        return {
            (m, effect, rep): self.simulate(m, effect, rep)
            for effect in self.effects
            for m in self.m
            for rep in range(self.cfg.reps)
        }

    # -------------------------------------------------------------------------
    # MB

    def sampleMB(self) -> None:
        cfg, priors, formula = self.cfg, self.design['priors'], self.design['formula']
        device = torch.device(cfg.device)
        t0 = time.perf_counter()
        mb = Api.from_pretrained(self.design['family'], device=device, verbose=False)
        load_s = time.perf_counter() - t0

        trials = self.trials()
        for name, prior in priors.items():
            first = next(iter(trials.values()))
            checkTrainingRange(
                mb.prepareData(first, formula=formula, priors={name: prior}, stage='dataset')
            )

        # one call per effect, each timed end to end (preparation, chunked network passes,
        # refinement): the proposals stay on the GPU, and 20,000 binary rows exceed 80 GiB
        records, units, call_s = [], [], []
        for effect in self.effects:
            cell = {key: df for key, df in trials.items() if key[1] == effect}
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = mb.sample(cell, formula=formula, priors=priors, random_walk=True, seed=0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            call_s.append(time.perf_counter() - t0)
            print(f'sample: {len(cell)} trials x {len(priors)} priors in {call_s[-1]:.1f} s')
            records += [
                r for key, res in results.items() for r in self._record(mb, key, res, cell[key])
            ]
            if cfg.n_nuts > 0:
                units += self._nutsUnits(mb, cell, results)
            del results
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        table = pd.DataFrame(records)
        table.to_csv(self.outdir / 'mb.csv', index=False)
        rows = len(trials) * len(priors)
        timing = {
            'device': torch.cuda.get_device_name()
            if device.type == 'cuda'
            else platform.processor(),
            'torch': torch.__version__,
            'trials': len(trials),
            'rows': rows,
            'skipped': int(table['skipped'].sum()) // len(priors),
            'load_s': load_s,
            'call_s': call_s,
            'sample_s': sum(call_s),
            'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30
            if device.type == 'cuda'
            else None,
        }
        (self.outdir / 'timing.json').write_text(json.dumps(timing, indent=2))
        print(json.dumps(timing, indent=2))
        if units:
            self.exportNuts(units, sum(call_s) / rows)

    def _record(self, mb: Api, key: tuple, result, df: pd.DataFrame) -> list[dict]:
        m, effect, rep = key
        base = {'m': m, 'effect': effect, 'rep': rep}
        if self.lf == 0:  # the exact analysis: two-sample t-test on school means
            means = df.groupby('school').agg(score=('score', 'mean'), program=('program', 'first'))
            t = stats.ttest_ind(means.score[means.program == 1], means.score[means.program == 0])
            base['t_detect'] = bool(t.pvalue < 0.05)
        if result is None:
            return [base | {'prior': p, 'skipped': True} for p in self.design['priors']]
        if result.param_names['ffx'] != ['Intercept', 'program']:
            raise ValueError(f'unexpected fixed effects {result.param_names["ffx"]}')
        sg = result.safeguards
        out = []
        for row, entry in enumerate(result.validation):
            draws = mb.posteriorDraws(result, prior_id=row)['program']
            lo, hi = draws.quantile([0.025, 0.975])
            out.append(
                base
                | {
                    'prior': entry['prior_name'],
                    'skipped': False,
                    'lo': lo,
                    'hi': hi,
                    'mean': draws.mean(),
                    'sd': draws.std(),
                    'detect': bool(detects(lo, hi)),
                    'accept_rate': sg['accept_rate'][row],
                    'accept_flag': bool(sg['accept_flag'][row]),
                    'ess_min': sg['ess_min'][row],
                    'ess_flag': bool(sg['ess_flag'][row]),
                    'random_walk_ran': bool(sg['random_walk_ran'][row]),
                    'map_z': sg['map_z'][row],
                    'map_z_flag': bool(sg['map_z_flag'][row]),
                }
            )
        return out

    def _nutsUnits(self, mb: Api, trials: dict, results: dict) -> list[tuple]:
        """Per prior, the first n_nuts replicates of every cell: the unit, its fit.py dataset and
        MB's draws of it, global (S, d + q [+ 1]) and local (m, S, q)."""
        priors, formula = self.design['priors'], self.design['formula']
        out = []
        for (m, effect, rep), result in results.items():
            if rep >= self.cfg.n_nuts or result is None:
                continue
            for row, entry in enumerate(result.validation):
                name = entry['prior_name']
                ds = mb.prepareData(
                    trials[(m, effect, rep)],
                    formula=formula,
                    priors={name: priors[name]},
                    stage='dataset',
                )
                checkTrainingRange(ds)
                ds = {k: v for k, v in ds.items() if not k.startswith('_') and k != 'Z'}
                for key in ('ffx', 'sigma_rfx', 'rfx', 'corr_rfx'):  # unknown to the references
                    ds[key] = np.full_like(ds[key], np.nan, dtype=np.float64)
                # the proposal is padded to the checkpoint's d and q: keep the active columns
                p, batch = result.proposal, result.batch
                d_mask, q_mask = batch['mask_d'][row].bool(), batch['mask_q'][row].bool()
                parts = [p.ffx[row][:, d_mask], p.sigma_rfx[row][:, q_mask]]
                if p.has_sigma_eps:
                    parts.append(p.sigma_eps[row][:, None])
                draws_g = torch.cat(parts, -1).float().cpu().numpy()  # (S, d + q [+ 1])
                draws_l = p.rfx[row][:m][..., q_mask].float().cpu().numpy()  # (m, S, q)
                unit = {'m': m, 'effect': effect, 'rep': rep, 'prior': name}
                out.append((unit, ds, draws_g, draws_l))
        return out

    def exportNuts(self, units: list[tuple], mb_tpd: float) -> None:
        """The units as a fit.py batch, MB's draws of each in the same order (mb_draws.npz) and
        the units table (units.csv)."""
        table, datasets, draws_g, draws_l = zip(*units)
        local = np.zeros((len(draws_l), max(self.m), *draws_l[0].shape[1:]), dtype=np.float32)
        for i, d in enumerate(draws_l):
            local[i, : d.shape[0]] = d
        folder = self.batch_path.parent
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.batch_path, **aggregate(list(datasets)))
        config = {'data_id': folder.name, 'likelihood_family': self.lf}
        (folder / 'config.yaml').write_text(yaml.safe_dump(config))
        np.savez_compressed(
            folder / 'mb_draws.npz',
            samples_g=np.stack(draws_g),  # (N, S, D)
            samples_l=local,  # (N, m_max, S, q)
            tpd=np.array(mb_tpd),  # batched wall time per posterior
        )
        pd.DataFrame(table).rename_axis('idx').to_csv(folder / 'units.csv')
        print(f'exported {len(table)} NUTS units to {self.batch_path}')

    # -------------------------------------------------------------------------
    # MB vs NUTS

    def compare(self) -> None:
        """real_posterior.py's metrics on the exported batch, plus both methods' detections."""
        folder = self.batch_path.parent
        units = pd.read_csv(folder / 'units.csv')
        col = Collection(self.batch_path, permute=False, fits=(REFERENCE_TAG, TIMING_TAG))
        batch = collateGrouped([col[i] for i in range(len(col))])
        with np.load(folder / 'mb_draws.npz') as raw:
            g, local, mb_tpd = (raw[k] for k in ('samples_g', 'samples_l', 'tpd'))
        m_max = batch['mask_n'].shape[1]
        p_mb = Proposal(
            {
                'global': {'samples': torch.from_numpy(g)},  # (B, S, D)
                'local': {'samples': torch.from_numpy(local[:, :m_max])},  # (B, m, S, q)
            },
            has_sigma_eps=self.lf == 0,
        )

        conv = nutsConverged(batch, REFERENCE_TAG)
        units['converged'] = conv
        units['nuts_duration'] = batch[f'{TIMING_TAG}_duration'].numpy()
        units['nuts_level'] = batch[f'{TIMING_TAG}_level'].numpy()
        batch, p_mb = subsetBatch(batch, conv), subsetProposal(p_mb, conv)
        # real_posterior.py's default --rescale to the response scale; Api.sample has already
        # rescaled MB's draws (Api._rescaleProposal), the fit.py references are standardized
        p_nuts = fit2proposal(batch, REFERENCE_TAG)
        p_nuts.rescale(batch['sd_y'])
        batch = rescaleData(batch)

        loo = {}
        for label, p in (('mb', p_mb), ('nuts', p_nuts)):
            summary = getSummary(
                p, batch, likelihood_family=self.lf, compute_pred_coverage=False,
                dataset_chunk_size=4,
            )  # fmt: skip
            loo[label] = summary.per_dataset.loo_nll.float().numpy()
            q = torch.quantile(p.ffx[..., 1].float(), torch.tensor([0.025, 0.975]), dim=-1)
            units.loc[conv, f'{label}_lo'] = q[0].numpy()
            units.loc[conv, f'{label}_hi'] = q[1].numpy()
            units.loc[conv, f'{label}_detect'] = detects(q[0].numpy(), q[1].numpy())
        units.loc[conv, 'r'] = computeCorr(p_mb, p_nuts, batch)
        units.loc[conv, 'sigma_ratio'] = computeSigmaRatio(p_mb, p_nuts, batch)
        units.loc[conv, 'rank_mad'] = computeRankMAD(p_mb, p_nuts, batch)
        units.loc[conv, 'delta_nll'] = loo['mb'] - loo['nuts']
        units['delta_tpd'] = float(mb_tpd) - units['nuts_duration']
        units.to_csv(self.outdir / 'compare.csv', index=False)
        print(f'{conv.sum()}/{len(conv)} units with a converged {REFERENCE_TAG}')

    # -------------------------------------------------------------------------
    # Report

    def report(self) -> None:
        mb = pd.read_csv(self.outdir / 'mb.csv')
        timing = json.loads((self.outdir / 'timing.json').read_text())
        lines = [f'# Power analysis: {self.name}', '', f'MB call: `{json.dumps(timing)}`', '']

        fitted = mb[~mb.skipped]
        cells = fitted.groupby(['effect', 'm', 'prior'], sort=True)
        table = cells.agg(
            trials=('detect', 'size'),
            power=('detect', 'mean'),
            moves_ran=('random_walk_ran', 'mean'),
            accept_flag=('accept_flag', 'mean'),
            map_z_flag=('map_z_flag', 'mean'),
        )
        table['mc_se'] = np.sqrt(table.power * (1 - table.power) / table.trials)
        table['skipped'] = mb[mb.skipped].groupby(['effect', 'm', 'prior']).size()
        table['skipped'] = table['skipped'].fillna(0).astype(int)
        if self.lf == 0:
            exact = exactPower(table.index.get_level_values('effect').to_numpy(),
                               self.design['icc'], table.index.get_level_values('m').to_numpy())  # fmt: skip
            table['exact'] = exact
            table['z_exact'] = (table.power - exact) / table.mc_se
            paired = fitted.assign(diff=fitted.detect.astype(float) - fitted.t_detect)
            diff = paired.groupby(['effect', 'm', 'prior'])['diff']
            table['t_power'] = cells['t_detect'].mean()
            table['mb_minus_t'] = diff.mean()
            table['mb_minus_t_se'] = diff.std() / np.sqrt(diff.size())
        lines += ['## Power', '', table.round(4).to_markdown(), '']

        path = self.outdir / 'compare.csv'
        if path.exists():
            lines += self._agreement(pd.read_csv(path))
        (self.outdir / 'report.md').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))

    def _agreement(self, units: pd.DataFrame) -> list[str]:
        """Pass criterion (fixed 2026-11-07): per m, odds ratio and prior, the paired MB - NUTS
        difference in detection rate within 2 SE and >= 95% agreeing detections; type I error of
        the weakly informative prior in [0.025, 0.075] (Bradley, 1978). Plus real_posterior.py's
        metrics, median and MAD over units."""
        conv = units[units.converged].copy()
        conv['diff'] = conv.mb_detect.astype(float) - conv.nuts_detect.astype(float)
        conv['agree'] = conv.mb_detect == conv.nuts_detect
        cells = conv.groupby(['effect', 'm', 'prior'])
        table = cells.agg(
            units=('diff', 'size'),
            mb_power=('mb_detect', 'mean'),
            nuts_power=('nuts_detect', 'mean'),
            diff=('diff', 'mean'),
            agree=('agree', 'mean'),
        )
        table['diff_se'] = cells['diff'].std() / np.sqrt(table.units)
        table['diff_ok'] = table['diff'].abs() <= 2 * table.diff_se.fillna(0)
        table['agree_ok'] = table.agree >= 0.95
        table['not_converged'] = units[~units.converged].groupby(['effect', 'm', 'prior']).size()
        table['not_converged'] = table['not_converged'].fillna(0).astype(int)

        mb = pd.read_csv(self.outdir / 'mb.csv')
        null = mb[~mb.skipped & (mb.effect == 1.0)] if self.lf == 1 else mb.iloc[:0]
        type1 = null.groupby(['m', 'prior']).detect.mean().rename('type_I')
        bradley = type1.index.get_level_values('prior').isin(self.design.get('bradley', ()))
        type1_ok = type1.between(0.025, 0.075) | ~bradley

        metrics = ['r', 'sigma_ratio', 'rank_mad', 'delta_nll', 'delta_tpd']
        rows = []
        for keys, group in units[units.converged].groupby(['prior', 'm']):
            stats_ = {k: _medianMad(group[k].to_numpy(dtype=float)) for k in metrics}
            rows.append(dict(zip(('prior', 'm'), keys)) | {
                k: f'{med:.3f} ± {mad:.3f}' for k, (med, mad) in stats_.items()
            })  # fmt: skip
        passed = table.diff_ok.all() and table.agree_ok.all() and type1_ok.all()
        return [
            '## MB vs NUTS (nuts2, converged units)', '',
            table.round(4).to_markdown(), '',
            '## Type I error (OR 1)', '', type1.round(4).to_frame().assign(ok=type1_ok).to_markdown(), '',
            "## real_posterior.py's metrics (median ± MAD)", '',
            pd.DataFrame(rows).to_markdown(index=False), '',
            f'**Pass criterion:** {"passed" if passed else "FAILED"}', '',
        ]  # fmt: skip

    # -------------------------------------------------------------------------
    # Figure

    def figure(self) -> None:
        """Detection rate vs schools, one panel per prior: MB on every trial (line, band of 2
        Monte Carlo SE), NUTS on the subsample where nuts2 converged (points, 2 SE)."""
        mb = pd.read_csv(self.outdir / 'mb.csv')
        mb = mb[~mb.skipped]
        path = self.outdir / 'compare.csv'
        nuts = pd.read_csv(path) if path.exists() else None
        priors = self.design['priors']
        null = 1.0 if self.lf == 1 else None  # the odds ratio of the type I error trials

        def rate(detected: pd.Series) -> tuple[pd.Series, pd.Series]:
            p = detected.groupby(level='m').agg(['mean', 'size'])
            return p['mean'], np.sqrt(p['mean'] * (1 - p['mean']) / p['size'])

        with plt.rc_context(FIG_RC):
            fig, axes = plt.subplots(
                1, len(priors), figsize=(FIG_W, PANEL_H + LEGEND_H), sharey=True, squeeze=False
            )
            for ax, (name, prior) in zip(axes[0], priors.items()):
                for effect in self.effects:
                    ls, fill = ('--', 'white') if effect == null else ('-', C_MB)
                    sel = mb[(mb.prior == name) & (mb.effect == effect)].set_index('m')
                    power, se = rate(sel.detect)
                    ax.plot(power.index, power, color=C_MB, ls=ls, lw=1.6)
                    ax.fill_between(
                        power.index, power - 2 * se, power + 2 * se, color=C_MB, alpha=0.13, lw=0
                    )
                    if nuts is not None:
                        sel = nuts[nuts.converged & (nuts.prior == name) & (nuts.effect == effect)]
                        p, se = rate(sel.set_index('m').nuts_detect.astype(float))
                        mfc = 'white' if effect == null else C_NUTS
                        ax.errorbar(
                            p.index, p, yerr=2 * se, fmt='o', color=C_NUTS, mfc=mfc, ms=4.5, lw=1
                        )
                    if self.lf == 1 and ax is axes[0, -1]:
                        label = (
                            'type I error (OR 1)' if effect == null else f'power (OR {effect:g})'
                        )
                        y = 0.12 if effect == null else power.iloc[-1] - 0.12
                        ax.text(self.m[-1], y, label, ha='right', va='center', color='0.25')
                for level in (0.05, 0.8):
                    ax.axhline(level, color='0.6', lw=0.8, ls=':', zorder=0)
                title = 'default priors' if prior is None else (
                    f'{name} prior, program ~ N(0, {prior["fixed"]["program"]["tau"]:g})'
                )  # fmt: skip
                ax.set(title=title, xlabel='schools per trial', xticks=self.m, ylim=(0, 1))
            axes[0, 0].set_ylabel('trials detecting the effect')
            fig.legend(
                [Line2D([], [], color=C_MB, lw=1.6), Line2D([], [], color=C_NUTS, marker='o', ls='')],
                ['MB, every trial', 'NUTS, subsample'],
                loc='lower center', ncol=2, frameon=False, handlelength=1.4, columnspacing=1.0,
            )  # fmt: skip
            fig.tight_layout(rect=(0, LEGEND_H / (PANEL_H + LEGEND_H), 1, 1))
            out = self.outdir / f'power_{self.name}.pdf'
            fig.savefig(out, bbox_inches='tight', pad_inches=0.15, dpi=VECTOR_RASTER_DPI)
        print(f'wrote {out}')


# =============================================================================
# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulation-based power analysis: MB vs NUTS')
    parser.add_argument('--design', type=str, required=True, choices=list(DESIGNS))
    parser.add_argument('--stage', type=str, required=True, choices=['mb', 'compare', 'report', 'figure'])
    parser.add_argument('--name', type=str, default=None, help='Run name for outputs (default: the design)')
    parser.add_argument('--reps', type=int, default=1000, help='Trials per cell')
    parser.add_argument('--m', type=int, nargs='+', default=None, help='Numbers of schools (default: the design)')
    parser.add_argument('--effects', type=float, nargs='+', default=None, help='Effects (default: the design)')
    parser.add_argument('--n_nuts', type=int, default=0, help='Replicates per cell exported for NUTS')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--outdir', type=str, default=str(RESULTS_DIR / 'power_analysis'))
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    cfg = setup()
    analysis = PowerAnalysis(cfg)
    {
        'mb': analysis.sampleMB,
        'compare': analysis.compare,
        'report': analysis.report,
        'figure': analysis.figure,
    }[cfg.stage]()
