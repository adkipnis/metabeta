"""Time-accuracy curve of the reference methods and metabeta, plus the false-convergence check.

Every test set carries the whole reference ladder: NUTS at three budgets (``nuts0`` <
``nuts1`` < ``nuts2``, PyMC defaults first), ADVI after 10k and 100k iterations (``advi0``,
``advi1``), Pathfinder with 4 and 20 paths (``pathfinder0``, ``pathfinder1``), Laplace, and
the composite ``nuts`` (cheapest converged level per dataset, cumulative wall time). This
script scores each of them, together with MB^0 (raw flow) and MB (flow + IMH), against the
true parameters on the datasets where the ``nuts2`` reference converged, pairs every score
with the method's median wall time, and draws the curve (x = wall time, y = metric).
metabeta's wall time is the per-dataset latency measured by runtimes.py (batch of one): the
CPU series is timed here when uncached, the GPU series is read from the ``*_cuda.json``
runtime cache next to the data when one has been pulled from the cluster.

The false-convergence check compares every NUTS / ADVI / Pathfinder level and Laplace to the
``nuts2`` reference with the paper's paired agreement metrics (real_posterior.py: r of the
posterior means, sigma-ratio, rank-MAD). NUTS levels are split by their own diagnostics, so
the table answers whether a run that passes the criterion also agrees with the reference and
how far a run that fails it is off.

Usage (from repo root):
    uv run python experiments/evaluation/ref_curve.py --families n --sizes small
    uv run python experiments/evaluation/ref_curve.py --ds_types sampled real
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from metabeta.utils.dataloader import subsetBatch
from metabeta.utils.device import setDevice
from metabeta.utils.evaluation import nutsConvergeMask, subsetProposal
from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, RESULTS_DIR
from metabeta.utils.fits import fitPath
from metabeta.utils.logger import setupLogging
from metabeta.utils.plot import DPI, savePlot
from metabeta.utils.posterior_eval import (
    fit2proposal,
    fitBatchMask,
    loadModel,
    loadOrComputeSummary,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
    validMethods,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402
from oracle_posterior import (  # noqa: E402
    REFERENCE_TAG,
    _capFull,
    flattenActiveParams,
    loadRegimeBatch,
    methodFitBatch,
    nutsConvergeMaskFromNpz,
)
from real_posterior import computeCorr, computeRankMAD, computeSigmaRatio  # noqa: E402
from runtimes import (
    FAMILY_NAMES,
    MB_DEFAULT,
    MB_FLOW,
    cachePath,
    collectCell,
    loadCache,
)  # noqa: E402

logger = logging.getLogger(__name__)

# =============================================================================
# Methods

# competitor tags in ladder order; the composite `nuts` is a user's escalation strategy
FIT_TAGS = ('nuts0', 'nuts1', 'nuts', 'advi0', 'advi1', 'pathfinder0', 'pathfinder1', 'laplace')
LADDERS = {
    'NUTS': ('nuts0', 'nuts1'),
    'ADVI': ('advi0', 'advi1'),
    'PF': ('pathfinder0', 'pathfinder1'),
}
AGREEMENT_TAGS = ('nuts0', 'nuts1', 'advi0', 'advi1', 'pathfinder0', 'pathfinder1', 'laplace')
LABELS = {
    'nuts0': 'NUTS L0',
    'nuts1': 'NUTS L1',
    'nuts': 'NUTS',
    'advi0': 'ADVI 10k',
    'advi1': 'ADVI 100k',
    'pathfinder0': 'PF 4',
    'pathfinder1': 'PF 20',
    'laplace': 'LA',
    'mb0_cpu': 'MB$^0$ (CPU)',
    'mb_cpu': 'MB (CPU)',
    'mb0_cuda': 'MB$^0$ (GPU)',
    'mb_cuda': 'MB (GPU)',
}
METRICS = ('NRMSE', 'EACE', 'LOO-NLL')
AGREEMENT = ('r', 'sigma_ratio', 'rank_mad')


# =============================================================================
class CurveExperiment:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.device = setDevice(cfg.device)
        self.outdir = Path(cfg.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []  # data_id, method, metric, value
        self.agreement: list[dict] = []

    # -------------------------------------------------------------------------
    # Per data id

    def go(self) -> None:
        for size in self.cfg.sizes:
            for family in self.cfg.families:
                for ds_type in self.cfg.ds_types:
                    self.collect(f'{size}-{family}-{ds_type}')
        if not self.rows:
            raise FileNotFoundError('no test set with a converged nuts2 reference was found')
        self.saveTables()
        self.plot()

    def collect(self, data_id: str) -> None:
        size, family, ds_type = data_id.split('-')
        data_path = DATA_DIR / data_id / 'test.npz'
        seed = BEST_SEEDS.get((FAMILY_NAMES[family], size))
        ckpt_dir = _ckpt_dir(FAMILY_NAMES[family], size, seed) if seed is not None else None
        if not data_path.exists() or ckpt_dir is None or not ckpt_dir.exists():
            logger.warning('%s: data or checkpoint missing, skipping', data_id)
            return
        if not fitPath(data_path, REFERENCE_TAG).exists():
            logger.warning('%s: no %s reference, skipping', data_id, REFERENCE_TAG)
            return
        logger.info('--- %s ---', data_id)

        model, model_cfg = loadModel(ckpt_dir, self.cfg.prefix, self.device)
        lf = int(model_cfg.likelihood_family)
        rescale = lf == 0
        data, n_total, n_kept, cap_mask = loadRegimeBatch(
            data_path, model_cfg.max_d, model_cfg.max_q
        )
        conv = nutsConvergeMaskFromNpz(data_path, cap_mask)  # over the capacity-kept datasets
        logger.info('  %s converged: %d / %d', REFERENCE_TAG, int(conv.sum()), n_kept)
        if not conv.any():
            return
        ctx = dict(
            data_id=data_id,
            data_path=data_path,
            ckpt_dir=ckpt_dir,
            lf=lf,
            rescale=rescale,
            cap_mask=cap_mask,
            conv=conv,
            max_d=model_cfg.max_d,
            max_q=model_cfg.max_q,
        )

        # metabeta: accuracy from the (cached) posterior samples, wall time from runtimes.py
        mb0, _ = loadOrSampleMB(
            model,
            data,
            data_path,
            ckpt_dir,
            self.cfg.prefix,
            self.cfg.n_samples,
            self.cfg.batch_size,
            self.cfg.seed,
            self.device,
            cap_mask,
        )
        if rescale:
            mb0.rescale(data['sd_y'])
            data = rescaleData(data)
        refine = validMethods(posthocDefaults(lf), lf)[0]
        mb, _ = loadOrRefine(
            refine,
            mb0,
            data,
            data_path,
            ckpt_dir,
            self.cfg.prefix,
            self.cfg.n_samples,
            self.cfg.seed,
            lf,
            rescale,
            cap_mask,
            self.cfg.batch_size,
            device=self.device,
        )
        ctx['data'] = data
        data_conv = subsetBatch(data, conv)
        latency = self._latencies(ctx, family, size, ds_type)
        for name, proposal, method in (('mb0', mb0, 'mb'), ('mb', mb, refine)):
            summary = self._summary(
                ctx, subsetProposal(proposal, conv), data_conv, method, conv, True
            )
            for device, times in latency[name].items():
                self._addRows(ctx, f'{name}_{device}', summary, data_conv, times[conv])
        del mb0, mb

        # reference methods: one fit file resident at a time
        reference: Proposal | None = None
        ref_success = None
        for tag in (REFERENCE_TAG, *FIT_TAGS):
            if not fitPath(data_path, tag).exists():
                logger.info('  %s: no fit file, skipping', tag)
                continue
            fit_batch = loadRegimeBatch(data_path, ctx['max_d'], ctx['max_q'], fits=(tag,))[0]
            success = fitBatchMask(fit_batch, tag) & conv
            proposal = fit2proposal(subsetBatch(methodFitBatch(fit_batch, tag), success), tag)
            batch_sub = subsetBatch(data, success)
            if rescale:
                proposal.rescale(batch_sub['sd_y'])
            times = fit_batch[f'{tag}_duration'].numpy()[success]
            if tag == REFERENCE_TAG:
                reference, ref_success = proposal, success
            else:
                summary = self._summary(ctx, proposal, batch_sub, tag, success, False)
                self._addRows(ctx, tag, summary, batch_sub, times)
            if tag in AGREEMENT_TAGS:
                own = nutsConvergeMask(fit_batch, tag)[success] if tag.startswith('nuts') else None
                self._addAgreement(ctx, tag, proposal, success, reference, ref_success, own)
            del fit_batch, proposal

    def _latencies(self, ctx: dict, family: str, size: str, ds_type: str) -> dict:
        """Per-dataset MB^0 / MB latency over the capacity-kept datasets, per device."""
        cfg = argparse.Namespace(
            ds_type=ds_type,
            prefix=self.cfg.prefix,
            n_samples=self.cfg.n_samples,
            seed=self.cfg.seed,
            rescale=True,
            method=None,
            max_datasets=None,
            refresh_cache=False,
        )
        out = {'mb0': {}, 'mb': {}}
        n_total = len(ctx['cap_mask'])
        records = collectCell(cfg, family, size, self.device) or []
        for method, key in ((MB_FLOW, 'mb0'), (MB_DEFAULT, 'mb')):
            arr = np.full(n_total, np.nan)
            for r in records:
                if r['method'] == method:
                    arr[r['idx']] = r['duration']
            out[key][self.device.type] = arr[ctx['cap_mask']]
        for other in ('cuda',) if self.device.type != 'cuda' else ():
            cache = loadCache(
                cachePath(
                    ctx['data_path'],
                    ctx['ckpt_dir'],
                    self.cfg.prefix,
                    self.cfg.n_samples,
                    self.cfg.seed,
                    torch.device(other),
                ),
                ctx['data_path'],
                ctx['ckpt_dir'],
                self.cfg.prefix,
            )
            if not cache:
                continue
            refine = validMethods(posthocDefaults(ctx['lf']), ctx['lf'])[0]
            for key, tag in (('mb0', 'flow'), ('mb', f'mb_{refine}_rs1')):
                arr = np.array([cache.get(f'{tag}:{i}', np.nan) for i in range(n_total)])
                out[key][other] = arr[ctx['cap_mask']]
        return out

    def _summary(self, ctx, proposal, batch, method, mask_sub, model_derived):
        proposal.to('cpu')
        return loadOrComputeSummary(
            proposal,
            batch,
            ctx['data_path'],
            method,
            _capFull(ctx['cap_mask'], mask_sub),
            ctx['lf'],
            ctx['rescale'],
            ckpt_dir=ctx['ckpt_dir'] if model_derived else None,
            prefix=self.cfg.prefix if model_derived else None,
            n_samples=self.cfg.n_samples if model_derived else None,
            seed=self.cfg.seed if model_derived else None,
            summary_chunk_size=self.cfg.summary_chunk_size,
        )

    def _addRows(self, ctx, method, summary, batch, times) -> None:
        ag = summary.aggregated
        active_d, active_q = batch['mask_d'].any(0), batch['mask_q'].any(0)
        has_eps = 'sigma_eps' in ag.nrmse
        values = {
            'NRMSE': flattenActiveParams(ag.nrmse, active_d, active_q, has_eps).median(),
            'EACE': flattenActiveParams(ag.eace, active_d, active_q, has_eps).median(),
            'LOO-NLL': summary.per_dataset.mloonll,
            'time': float(np.nanmedian(times)),
            'time_p90': float(np.nanpercentile(times, 90)),
            'n': int(len(times)),
        }
        for metric, value in values.items():
            self.rows.append(
                {
                    'data_id': ctx['data_id'],
                    'method': method,
                    'metric': metric,
                    'value': float(value),
                }
            )
        logger.info(
            '  %-12s NRMSE %.3f  EACE %.3f  LOO-NLL %s  time %.1fs (n=%d)',
            LABELS.get(method, method),
            values['NRMSE'],
            values['EACE'],
            f"{values['LOO-NLL']:.3f}" if values['LOO-NLL'] is not None else 'NA',
            values['time'],
            values['n'],
        )

    def _addAgreement(self, ctx, tag, proposal, success, reference, ref_success, own) -> None:
        """Paired agreement of ``tag`` with the nuts2 reference on their common datasets."""
        common = success & ref_success  # over the capacity-kept datasets
        p = subsetProposal(proposal, common[success])
        ref = subsetProposal(reference, common[ref_success])
        batch = subsetBatch(ctx['data'], common)  # the metrics read only its masks
        metrics = {
            'r': computeCorr(p, ref, batch),
            'sigma_ratio': computeSigmaRatio(p, ref, batch),
            'rank_mad': computeRankMAD(p, ref, batch),
        }
        groups = {'all': np.ones(int(common.sum()), bool)}
        if own is not None:
            own = own[common[success]]
            groups = {'converged': own, 'not converged': ~own}
        for group, sel in groups.items():
            if not sel.any():
                continue
            row = {'data_id': ctx['data_id'], 'method': tag, 'group': group, 'n': int(sel.sum())}
            row.update({k: float(np.nanmedian(v[sel])) for k, v in metrics.items()})
            self.agreement.append(row)

    # -------------------------------------------------------------------------
    # Output

    def saveTables(self) -> None:
        stem = self.outdir / f'ref_curve_{self.cfg.tag}'
        lines = ['data_id,method,metric,value']
        lines += [f"{r['data_id']},{r['method']},{r['metric']},{r['value']:.6g}" for r in self.rows]
        (stem.with_suffix('.csv')).write_text('\n'.join(lines) + '\n')

        header = ['data id', 'method', 'group', 'n', 'r ↑', 'σ-ratio → 1', 'rank-MAD ↓']
        md = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        for r in self.agreement:
            md.append(
                f"| {r['data_id']} | {LABELS.get(r['method'], r['method'])} | {r['group']} | {r['n']} "
                f"| {r['r']:.3f} | {r['sigma_ratio']:.3f} | {r['rank_mad']:.3f} |"
            )
        text = (
            f'# Agreement with the {REFERENCE_TAG} reference (median over datasets)\n\n'
            'Paired metrics of real_posterior.py on the datasets where both the method and '
            f'{REFERENCE_TAG} succeeded and {REFERENCE_TAG} converged; NUTS levels are split by '
            'their own convergence diagnostics (the false-convergence check).\n\n'
            + '\n'.join(md)
            + '\n'
        )
        Path(f'{stem}_agreement.md').write_text(text)
        logger.info('Wrote %s.csv and %s_agreement.md', stem, stem)

    def plot(self) -> None:
        # the curve scores against the true parameters, which only the sampled sets carry
        rows = [r for r in self.rows if r['data_id'].endswith('-sampled')]
        methods = [m for m in (*LABELS,) if any(r['method'] == m for r in rows)]
        plt.rc('font', size=13)  # the figure is set at text width in the paper
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.2), dpi=DPI)
        cmap = plt.get_cmap('tab10')
        colors = {m: cmap(i % 10) for i, m in enumerate(methods)}
        for ax, metric in zip(axes, METRICS):
            agg = {}
            for m in methods:
                t = [r['value'] for r in rows if r['method'] == m and r['metric'] == 'time']
                v = [r['value'] for r in rows if r['method'] == m and r['metric'] == metric]
                ax.scatter(t, v, color=colors[m], alpha=0.35, s=18, edgecolors='none')
                agg[m] = (float(np.median(t)), float(np.median(v)))
                ax.scatter(
                    *agg[m], color=colors[m], s=70, marker='o', label=LABELS.get(m, m), zorder=3
                )
            for tags in LADDERS.values():
                pts = [agg[t] for t in tags if t in agg]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color='0.4', lw=1.0, zorder=2)
            ax.set_xscale('log')
            if metric == 'LOO-NLL':  # a few diverged ADVI 10k sets would otherwise set the ceiling
                ax.set_yscale('log')
            ax.set_xlabel('wall time per dataset [s]')
            ax.set_ylabel(metric)
        axes[-1].legend(fontsize=10, frameon=False, loc='best')
        fig.tight_layout()
        path = savePlot(self.outdir, f'ref_curve_{self.cfg.tag}', ending='pdf')
        plt.close(fig)
        logger.info('Wrote %s', path)


# =============================================================================
# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Time-accuracy curve of the reference methods and metabeta.')
    parser.add_argument('--families', type=str, nargs='+', default=list(FAMILY_NAMES), choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=['small', 'medium', 'large', 'huge'])
    parser.add_argument('--ds_types', type=str, nargs='+', default=['sampled', 'real'])
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=4000, help='MB draws (the competitors carry 4000)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--tag', type=str, default='all', help='output stem suffix')
    parser.add_argument('--verbosity', type=int, default=1)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    experiment = CurveExperiment(cfg)
    experiment.go()
