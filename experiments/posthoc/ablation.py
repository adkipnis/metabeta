"""Post-hoc benchmark: all refinement methods on a common dataset subset.

Conditions (Normal)
-------------------
raw          : raw flow samples
is           : global IS with PSIS
isFull       : joint IS with PSIS — full=True adds the rfx prior and local log q to the
               weight, so it targets the exact joint posterior; expected to degenerate as
               the rfx dimension (m × q) grows, but is a valid candidate now that the
               2026-07-28 log-det fix makes log q_l trustworthy
isMarginal   : Rao-Blackwellised marginal SNIS (Normal only) — exact marginal weights with
               correlated Σ_rfx + LKJ prior, rfx redrawn from the exact conditional
isLaplace    : Laplace analog of isMarginal (Bernoulli/Poisson only) — nAGQ=1-style
               approximate marginal weights + Laplace-Gaussian conditional rfx redraw
rbAttach     : rfx attachment only (Bernoulli/Poisson only) — uniform weights, flow rfx
               replaced by Laplace conditional draws (zero weight bias)
imhMarginal  : IMH mode='marginal' (Normal, Rao-Blackwellised) or 'global' (other). Note:
               'global' still degrades calibration vs raw/is on non-Normal likelihoods (it
               never MH-corrects rfx — see metabeta/posthoc/metropolis.py's TODO); 'joint'
               was tried too and was worse (acceptance collapses with many groups). Neither
               is a real fix — see metropolis.py's "Findings" docstring section for the
               root-cause analysis (argmax chain init, inconsistent marginal target).
               NB: those findings predate the 2026-07-28 log-det fix (log q entered every
               IMH weight), so IMH is being re-evaluated in the post-fix ablation.
imhGlobal    : IMH mode='global' on Normal (same biased pseudo-target as 'is'); for
               non-Normal families this is already what imhMarginal falls back to, so the
               condition is Normal-only.
svgd         : SVGD with per-dim bandwidth + cosine LR decay — opt in with --include-svgd,
               off by default (too slow to be practically useful: ~40s/dataset, and gives
               a fraction of a nat of marginal-log-p improvement over the flow samples it starts
               from — see metabeta/outputs/results/ablation/*.md for reference numbers)
coldNuts     : NUTS results extracted from test.fit.npz (pre-computed, no rerun) — only
               available with --split=test
warmNuts     : warm-started NUTS (flow samples initialise PyMC chains) — opt in with
               --include-warmnuts, off by default (slow)

Note: 'laplace'/'laplaceIS' and 'cd' (coordinate descent) conditions were removed
along with metabeta/posthoc/laplace.py and metabeta/posthoc/coordinate.py, which
were deprecated in 3c5b7af2.

By default this evaluates every (family, size) combination in BEST_SEEDS on the
*entire* validation split (valid.npz), capped by --n-datasets if given. Run
functions are imported from the individual eval scripts; see those files for
per-method settings (e.g. IMH chains/steps, NUTS tune/draws).

Data loading uses Collection + collateGrouped so that per-dataset unpadded
dicts are available for NUTS-based methods.

Results are printed to stdout and also written as markdown to
metabeta/outputs/results/ablation/{family}_{size}.md (one file per model).

Run from repo root:
    uv run python experiments/posthoc/ablation.py
    uv run python experiments/posthoc/ablation.py --sizes small --families normal bernoulli
    uv run python experiments/posthoc/ablation.py --n-datasets 32 --include-svgd --include-warmnuts
"""

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import torch

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT

# Make benchmark method wrappers importable without installing benchmarks as a package.
sys.path.insert(0, str(REPO_ROOT / 'benchmarks'))
# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from eval_imh import runIMH, N_SAMPLES as IMH_N_SAMPLES               # noqa: E402
from eval_svgd import runSVGD                                          # noqa: E402
from eval_warmnuts import runWarmNuts                                  # noqa: E402
from build_ckpt import BEST_SEEDS, _ckpt_dir                           # noqa: E402

from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.models.approximator import Approximator
from metabeta.utils.evaluation import EvaluationSummary
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.posthoc.warmnuts import _stackProposals
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice
from metabeta.utils.results import Proposal, concatProposalsBatch
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.padding import unpad
from metabeta.utils.posterior_cache import loadProposalCache
from metabeta.utils.preprocessing import rescaleData

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
FAMILY_INITIAL = {'normal': 'n', 'bernoulli': 'b', 'poisson': 'p'}
LIKELIHOOD_FAMILY = {'normal': 0, 'bernoulli': 1, 'poisson': 2}
RESULTS_DIR = REPO_ROOT / 'metabeta' / 'outputs' / 'results' / 'ablation'


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', nargs='+', default=['small', 'medium'], choices=['small', 'medium', 'large', 'huge'], help='data/model sizes to evaluate')
    p.add_argument('--families', nargs='+', default=['normal', 'bernoulli', 'poisson'], choices=['normal', 'bernoulli', 'poisson'], help='likelihood families to evaluate')
    p.add_argument('--split', choices=['valid', 'test'], default='valid', help='npz split to evaluate on; only "test" has coldNuts fits')
    p.add_argument('--prefix', type=str, default='best', help='checkpoint prefix to load and to match evaluate.py MB caches against')
    p.add_argument('--device', type=str, default='cpu', help='device for flow sampling (posthoc methods and summaries stay on cpu)')
    p.add_argument('--batch-size', type=int, default=4, help='sub-batch size for torch-based methods')
    p.add_argument('--n-datasets', type=int, default=None, help='cap on datasets per model (default: use the entire split)')
    p.add_argument('--n-samples', type=int, default=1000, help='flow samples for torch-based methods (raw/is/svgd); IMH uses its own fixed count')
    p.add_argument('--skip', nargs='+', default=[], choices=['raw', 'is', 'isFull', 'isMarginal', 'isLaplace', 'rbAttach', 'imhMarginal', 'imhGlobal', 'svgd', 'coldNuts', 'warmNuts'], help='conditions to skip (e.g. --skip is)')
    p.add_argument('--include-svgd', action='store_true', help='also run the (slow) SVGD condition')
    p.add_argument('--include-warmnuts', action='store_true', help='also run the (slow) warm-started NUTS condition')
    return p.parse_args()
# fmt: on


def buildModels(families: list[str], sizes: list[str], prefix: str) -> list[dict]:
    models = []
    for family in families:
        for size in sizes:
            seed = BEST_SEEDS.get((family, size))
            if seed is None:
                print(f'[SKIP] no BEST_SEEDS entry for ({family}, {size})')
                continue
            models.append(
                dict(
                    label=f'{family.capitalize()} ({size})',
                    family=family,
                    size=size,
                    ckpt=_ckpt_dir(family, size, seed) / f'{prefix}.pt',
                    data_dir=DATA_DIR / f'{size}-{FAMILY_INITIAL[family]}-sampled',
                    likelihood_family=LIKELIHOOD_FAMILY[family],
                )
            )
    return models


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def loadModel(ckpt: Path) -> tuple[Approximator, int]:
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    model_cfg = ApproximatorConfig(**payload['model_cfg'])
    model = Approximator(model_cfg)
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model, payload['epoch']


def loadData(path: Path, n_limit: int | None) -> tuple[list, list[dict], dict, dict]:
    """Load up to n_limit datasets (or the entire split if n_limit is None).

    Returns
    -------
    items        : list of Collection items; used to build sub-batches
    ds_list      : list of fully-unpadded numpy dicts for NUTS / buildPymc
    tensor_batch : single collated tensor dict, un-rescaled (for NUTS flow init)
    full_batch   : rescaled tensor_batch (ground truth for evaluation)
    """
    # exclude the fitted posterior-sample arrays: a *.fit.npz decompresses to ~20 GB
    # (nuts_/advi_/laplace_ rfx are ~6.5 GB each); only runNutsFromNpz needs the nuts
    # arrays and it re-opens the file lazily itself
    col = Collection(path, permute=False, exclude_prefixes=('nuts_', 'advi_', 'laplace_'))
    n = len(col) if n_limit is None else min(n_limit, len(col))
    items = [col[i] for i in range(n)]
    tensor_batch = collateGrouped(items)

    ds_list = []
    for i in range(n):
        ds = {k: v[i] for k, v in col.raw.items()}
        sizes = {k: ds[k] for k in 'dqmn'}
        ds_list.append(unpad(ds, sizes))

    return items, ds_list, tensor_batch, rescaleData(tensor_batch)


def collectProposals(
    model: Approximator,
    items: list,
    n_samples: int,
    batch_size: int,
    device: str = 'cpu',
) -> tuple[list, list]:
    """Draw rescaled flow proposals in sub-batches; return (proposals, batches).

    Sampling runs on ``device``; proposals and batches are returned on cpu, where the
    posthoc methods and summaries run.
    """
    proposals, batches = [], []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = collateGrouped(items[i : i + batch_size])
            proposal = model.estimate(toDevice(batch, device), n_samples=n_samples)
            proposal.to('cpu')
            proposal.rescale(batch['sd_y'])
            batch = rescaleData(batch)
            proposals.append(proposal)
            batches.append(batch)
    return proposals, batches


# ---------------------------------------------------------------------------
# Cached MB posterior samples (reuse evaluate.py's cache when available)
# ---------------------------------------------------------------------------


def buildBatches(items: list, batch_size: int) -> list:
    """Rescaled data sub-batches (model-independent), aligned with the proposal sub-batches."""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(rescaleData(collateGrouped(items[i : i + batch_size])))
    return batches


def findMbCache(
    data_dir: Path, split: str, run_name: str, n_needed: int, prefix: str
) -> Path | None:
    """Smallest MB posterior-sample cache with >= n_needed samples for this checkpoint/split.

    Matches the full-split caches written by evaluate.py (partition.mb.{run}_{prefix}
    _s{n}_seed{s}_k{k}.npz); ignores masked-subset caches (e.g. real_posterior.py's
    ``test-<hash>.mb.*``) since those never match the ``{split}.mb.`` prefix.
    """
    best, best_s = None, None
    for path in data_dir.glob(f'{split}.mb.{run_name}_{prefix}_s*.npz'):
        match = re.search(r'_s(\d+)_seed\d+_k\d+\.npz$', path.name)
        if match is None:
            continue
        s = int(match.group(1))
        if s >= n_needed and (best_s is None or s < best_s):
            best, best_s = path, s
    return best


def _cacheFresh(cache: Path, data_path: Path, ckpt: Path) -> bool:
    """True if the cache is at least as new as the data file and the checkpoint."""
    ref_mtime = 0.0
    for ref in (data_path, ckpt):
        if ref.exists():
            ref_mtime = max(ref_mtime, ref.stat().st_mtime)
    return cache.stat().st_mtime >= ref_mtime


def _trimProposal(p: Proposal, n_samples: int, m: int) -> Proposal:
    """Trim a batch-sliced cached proposal to n_samples draws and m local groups.

    The merged cache pads local groups to the split-wide max_m; ``m`` is the sub-batch's
    own padding (from collateGrouped), so [:, :m] keeps every real group and drops only
    the extra split-wide padding. Flow draws are i.i.d., so the first n_samples are a
    valid subsample when the cache holds more than requested.
    """
    global_data = {'samples': p.samples_g[:, :n_samples, :].contiguous()}
    local_data = {'samples': p.samples_l[:, :m, :n_samples, :].contiguous()}
    if 'log_prob' in p.data['global']:
        global_data['log_prob'] = p.data['global']['log_prob'][:, :n_samples].contiguous()
    if 'log_prob' in p.data['local']:
        local_data['log_prob'] = p.data['local']['log_prob'][:, :m, :n_samples].contiguous()
    corr = p._corr_rfx
    if corr is not None:
        corr = corr[:, :n_samples].contiguous()
    trimmed = Proposal(
        {'global': global_data, 'local': local_data},
        has_sigma_eps=p.has_sigma_eps,
        d_corr=p.d_corr,
        corr_rfx=corr,
    )
    trimmed.reff = p.reff
    return trimmed


def splitMergedProposal(merged: Proposal, batches: list, n_samples: int) -> list:
    """Split a merged full-split cache into per-sub-batch proposals aligned with ``batches``."""
    proposals = []
    start = 0
    for batch in batches:
        b = batch['X'].shape[0]
        m = batch['rfx'].shape[1]
        sliced = merged.slice_b(start, start + b)
        proposals.append(_trimProposal(sliced, n_samples, m))
        start += b
    return proposals


def loadOrSampleProposals(
    model: Approximator,
    items: list,
    n_samples: int,
    data_dir: Path,
    split: str,
    run_name: str,
    data_path: Path,
    ckpt: Path,
    prefix: str,
    batch_size: int,
    device: str = 'cpu',
) -> tuple[list, list]:
    """Reuse evaluate.py's cached MB samples when a fresh, large-enough cache exists."""
    cache = findMbCache(data_dir, split, run_name, n_samples, prefix)
    if cache is not None and _cacheFresh(cache, data_path, ckpt):
        try:
            merged, _ = loadProposalCache(cache)
        except (KeyError, ValueError) as exc:
            # e.g. version-1 caches written before the 2026-07-28 log-det fix
            print(f'  MB samples (n={n_samples}): invalid cache {cache.name} ({exc}); sampling')
            return collectProposals(model, items, n_samples, batch_size, device)
        if merged.samples_g.shape[0] >= len(items):
            batches = buildBatches(items, batch_size)
            proposals = splitMergedProposal(merged, batches, n_samples)
            print(f'  MB samples (n={n_samples}): loaded cache {cache.name}')
            return proposals, batches
        print(
            f'  MB samples (n={n_samples}): cache {cache.name} has too few datasets '
            f'({merged.samples_g.shape[0]} < {len(items)}); sampling from model'
        )
    else:
        reason = 'no matching cache' if cache is None else f'stale cache {cache.name}'
        print(f'  MB samples (n={n_samples}): {reason}; sampling from model')
    return collectProposals(model, items, n_samples, batch_size, device)


def runRaw(proposals, full_batch, lf, summary_cache=None):
    if summary_cache is not None:
        print(summaryTable(summary_cache, lf))
        return
    proposal = concatProposalsBatch(proposals)
    print(summaryTable(getSummary(proposal, full_batch, likelihood_family=lf), lf))


def _printWeightHealth(proposal):
    k = proposal.pareto_k
    if k is None:
        return
    fallback = proposal.is_results.get('fallback')
    frac = fallback.float().mean().item() if fallback is not None else 0.0
    eff = proposal.efficiency
    eff_str = f'{eff.mean():.2f}' if eff is not None else 'n/a'
    print(
        f'  pareto_k mean={k.mean():.2f}  max={k.max():.2f}  '
        f'fallback={frac:.0%}  efficiency mean={eff_str}'
    )


def runIS(proposals, batches, full_batch, lf, full=False, marginal=False, rb_redraw=False):
    out = []
    with torch.no_grad():
        for p, batch in zip(proposals, batches):
            sampler = ImportanceSampler(
                batch,
                full=full,
                corr_prior=True,
                marginal=marginal,
                rb_redraw=rb_redraw,
                pareto=True,
                likelihood_family=lf,
            )
            # slice-copy so is_results / redrawn rfx don't mutate the shared proposals
            out.append(sampler(p.slice_b(0, p.samples_g.shape[0])))
    proposal = concatProposalsBatch(out)
    _printWeightHealth(proposal)
    print(summaryTable(getSummary(proposal, full_batch, likelihood_family=lf), lf))


def runISLaplace(proposals, batches, full_batch, lf, attach_only=False):
    out = []
    with torch.no_grad():
        for p, batch in zip(proposals, batches):
            sampler = LaplaceImportanceSampler(
                batch,
                attach_only=attach_only,
                corr_prior=True,
                pareto=True,
                likelihood_family=lf,
            )
            out.append(sampler(p.slice_b(0, p.samples_g.shape[0])))
    proposal = concatProposalsBatch(out)
    _printWeightHealth(proposal)
    print(summaryTable(getSummary(proposal, full_batch, likelihood_family=lf), lf))


def _nutsSummaryCache(npz_path: Path, n_ds: int) -> EvaluationSummary | None:
    """Cached full-split NUTS summary written by evaluate.py, if fresh and matching n_ds.

    Loading it skips the ~6.5 GB nuts_rfx materialization and the getSummary pass over
    NUTS's ~4000 draws, which otherwise dominates coldNuts runtime. Only returned when the
    cache covers exactly n_ds datasets (i.e. ablation runs the whole split, uncapped), so it
    stays comparable with the other conditions.
    """
    partition = npz_path.name.split('.')[0]
    cache_path = npz_path.parent / f'summary_{partition}_nuts.pt'
    if not cache_path.exists() or cache_path.stat().st_mtime < npz_path.stat().st_mtime:
        return None
    try:
        summary = EvaluationSummary.load(cache_path)
    except (KeyError, ValueError, RuntimeError):
        return None
    if summary.per_dataset.posterior_nll.shape[0] != n_ds:
        return None
    print(f'  [cached summary] loaded {cache_path.name}')
    return summary


def _mbSummaryCache(
    data_dir: Path,
    split: str,
    run_name: str,
    prefix: str,
    n_samples: int,
    n_ds: int,
    ckpt: Path,
) -> EvaluationSummary | None:
    """Cached MB (raw-flow) summary written by evaluate.py, if fresh and matching n_ds.

    The raw condition is just the MB checkpoint's flow samples, which evaluate.py already
    summarised — so reuse that summary instead of recomputing getSummary. Matched by
    checkpoint/prefix/n_samples (seed, k, and pred-coverage flag are globbed since ablation
    does not fix them); only returned for the full uncapped split (n_ds match).
    """
    pattern = f'summary_{split}_mb_{run_name}_{prefix}_s{n_samples}_seed*_k*_predcov*_all.pt'
    ref_mtime = ckpt.stat().st_mtime if ckpt.exists() else 0.0
    for cache_path in sorted(data_dir.glob(pattern)):
        if cache_path.stat().st_mtime < ref_mtime:
            continue
        try:
            summary = EvaluationSummary.load(cache_path)
        except (KeyError, ValueError, RuntimeError):
            continue
        if summary.per_dataset.posterior_nll.shape[0] != n_ds:
            continue
        print(f'  [cached summary] loaded {cache_path.name}')
        return summary
    return None


def runNutsFromNpz(npz_path: Path, ds_list: list, tensor_batch: dict, full_batch: dict, lf: int):
    """Extract pre-computed NUTS results from test.fit.npz and evaluate."""
    n_ds = len(ds_list)
    has_se = hasSigmaEps(lf)

    cached = _nutsSummaryCache(npz_path, n_ds)
    if cached is not None:
        # only the tiny diagnostic arrays are needed; skip the ~6.5 GB nuts_rfx load entirely
        with np.load(npz_path, allow_pickle=True) as data:
            nuts_divergences = data['nuts_divergences'][:n_ds]
            nuts_duration = data['nuts_duration'][:n_ds]
            nuts_ess = data['nuts_ess'][:n_ds]
            n_s = data['nuts_sigma_rfx'].shape[-1]
        total_divs = int(np.asarray(nuts_divergences).sum())
        total_time = float(np.asarray(nuts_duration, dtype=np.float64).sum())
        reff = float(nuts_ess.mean() / n_s)
        print(
            f'  divergences={total_divs}  reff={reff:.3f}  time/ds={total_time / n_ds:.1f}s  '
            f'(total={total_time:.0f}s)'
        )
        print(summaryTable(cached, lf))
        return

    # hoist each npz member once: NpzFile decompresses the *entire* member on every
    # [] access (nuts_rfx alone is ~6.5 GB), so per-iteration indexing re-decompresses
    # it n_ds times; slice to n_ds and downcast to float32 immediately
    with np.load(npz_path, allow_pickle=True) as data:
        nuts_ffx = data['nuts_ffx'][:n_ds].astype(np.float32)
        nuts_sigma_rfx = data['nuts_sigma_rfx'][:n_ds].astype(np.float32)
        nuts_sigma_eps = data['nuts_sigma_eps'][:n_ds].astype(np.float32) if has_se else None
        nuts_rfx = data['nuts_rfx'][:n_ds].astype(np.float32)
        nuts_corr_rfx = data['nuts_corr_rfx'][:n_ds].astype(np.float32)
        nuts_divergences = data['nuts_divergences'][:n_ds]
        nuts_duration = data['nuts_duration'][:n_ds]
        nuts_ess = data['nuts_ess'][:n_ds]

    proposals = []
    total_divs = 0
    total_time = 0.0

    n_s = nuts_ffx.shape[-1]
    for i, ds in enumerate(ds_list):
        d_i = int(ds['d'])
        q_i = int(ds['q'])
        m_i = int(ds['m'])

        ffx = torch.as_tensor(nuts_ffx[i, :d_i, :]).T                          # (n_s, d_i)
        sigma_rfx = torch.as_tensor(nuts_sigma_rfx[i, :q_i, :]).T              # (n_s, q_i)
        parts = [ffx, sigma_rfx]
        if has_se:
            sigma_eps = torch.as_tensor(nuts_sigma_eps[i, 0, :])
            parts.append(sigma_eps.unsqueeze(-1))                               # (n_s, 1)
        samples_g = torch.cat(parts, dim=-1).unsqueeze(0)                      # (1, n_s, D)

        rfx_raw = torch.as_tensor(nuts_rfx[i, :q_i, :m_i, :])                  # (q_i, m_i, n_s)
        samples_l = rfx_raw.permute(1, 2, 0).unsqueeze(0)                      # (1, m_i, n_s, q_i)

        corr_rfx = torch.as_tensor(
            nuts_corr_rfx[i, :, :, :q_i, :q_i]
        )                                                                        # (1, n_s, q_i, q_i)

        proposed = {
            'global': {'samples': samples_g, 'log_prob': torch.zeros(1, n_s)},
            'local': {'samples': samples_l, 'log_prob': torch.zeros(1, m_i, n_s)},
        }
        proposals.append(Proposal(proposed, has_sigma_eps=has_se, corr_rfx=corr_rfx))
        total_divs += int(nuts_divergences[i].sum())
        total_time += float(nuts_duration[i])

    reff = float(nuts_ess.mean() / n_s)
    print(
        f'  divergences={total_divs}  reff={reff:.3f}  time/ds={total_time / n_ds:.1f}s  '
        f'(total={total_time:.0f}s)'
    )

    target_d = tensor_batch['ffx'].shape[-1]
    target_q = tensor_batch['sigma_rfx'].shape[-1]
    merged = _stackProposals(proposals, target_d=target_d, target_q=target_q)
    merged.rescale(tensor_batch['sd_y'][:n_ds])
    merged.reff = reff
    print(summaryTable(getSummary(merged, full_batch, likelihood_family=lf), lf))


class _Tee:
    """Write to multiple streams at once (used to mirror stdout into a buffer)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _renderTerminal(text: str) -> str:
    """Collapse '\\r'-based progress overwrites (tqdm.write) to their final state."""
    return '\n'.join(line.split('\r')[-1] for line in text.split('\n'))


def main() -> None:
    args = setup()
    models = buildModels(args.families, args.sizes, args.prefix)

    conditions = [
        'raw',
        'is',
        'isFull',
        'isMarginal',
        'isLaplace',
        'rbAttach',
        'imhMarginal',
        'imhGlobal',
    ]
    if args.include_svgd:
        conditions.append('svgd')
    if args.split == 'test':
        conditions.append('coldNuts')
    if args.include_warmnuts:
        conditions.append('warmNuts')
    conditions = [c for c in conditions if c not in args.skip]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in models:
        lf = cfg['likelihood_family']
        buf = io.StringIO()
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            model, epoch = loadModel(cfg['ckpt'])
            model.to(args.device)
            print(f'\n{"#" * 70}')
            print(
                f'#  {cfg["label"]}  (epoch={epoch}  params={model.n_params:,}  '
                f'd_ffx={model.d_ffx}  d_rfx={model.d_rfx})'
            )
            print(f'{"#" * 70}')

            fit_npz = cfg['data_dir'] / 'test.fit.npz'
            split_name = 'test.fit.npz' if args.split == 'test' else 'valid.npz'
            data_path = cfg['data_dir'] / split_name
            items, ds_list, tensor_batch, full_batch = loadData(data_path, args.n_datasets)
            n_ds = len(ds_list)
            print(
                f'Datasets: {n_ds}  |  n_samples (flow-based): {args.n_samples}  |  '
                f'data: {data_path.name}'
            )
            print('(per-method settings: see individual eval_*.py benchmarks)\n')

            run_name = cfg['ckpt'].parent.name
            proposals, batches = loadOrSampleProposals(
                model,
                items,
                args.n_samples,
                cfg['data_dir'],
                args.split,
                run_name,
                data_path,
                cfg['ckpt'],
                args.prefix,
                args.batch_size,
                args.device,
            )
            # IMH requires exactly N_CHAINS × N_STEPS samples — draw a dedicated set.
            imh_proposals, imh_batches = loadOrSampleProposals(
                model,
                items,
                IMH_N_SAMPLES,
                cfg['data_dir'],
                args.split,
                run_name,
                data_path,
                cfg['ckpt'],
                args.prefix,
                args.batch_size,
                args.device,
            )

            for cond in conditions:
                if cond == 'isMarginal' and lf != 0:
                    continue  # exact marginal requires the Normal likelihood
                if cond in ('isLaplace', 'rbAttach') and lf == 0:
                    continue  # Normal has the exact marginal — Laplace is for GLMMs
                if cond == 'imhGlobal' and lf != 0:
                    continue  # non-Normal imhMarginal already runs mode='global'
                print('=' * 65)
                print(f'  {cond}')
                print('=' * 65)
                if cond == 'raw':
                    mb_summary = _mbSummaryCache(
                        cfg['data_dir'],
                        args.split,
                        run_name,
                        args.prefix,
                        args.n_samples,
                        n_ds,
                        cfg['ckpt'],
                    )
                    runRaw(proposals, full_batch, lf, summary_cache=mb_summary)
                elif cond == 'is':
                    runIS(proposals, batches, full_batch, lf)
                elif cond == 'isFull':
                    runIS(proposals, batches, full_batch, lf, full=True)
                elif cond == 'isMarginal':
                    runIS(proposals, batches, full_batch, lf, marginal=True, rb_redraw=True)
                elif cond == 'isLaplace':
                    runISLaplace(proposals, batches, full_batch, lf)
                elif cond == 'rbAttach':
                    runISLaplace(proposals, batches, full_batch, lf, attach_only=True)
                elif cond == 'imhMarginal':
                    imh_mode = 'marginal' if lf == 0 else 'global'
                    runIMH(imh_mode, imh_proposals, imh_batches, full_batch, lf)
                elif cond == 'imhGlobal':
                    runIMH('global', imh_proposals, imh_batches, full_batch, lf)
                elif cond == 'svgd':
                    runSVGD(proposals, batches, full_batch, lf)
                elif cond == 'coldNuts':
                    runNutsFromNpz(fit_npz, ds_list, tensor_batch, full_batch, lf)
                elif cond == 'warmNuts':
                    model.to('cpu')  # runWarmNuts feeds the cpu tensor_batch to model.estimate
                    runWarmNuts(model, tensor_batch, ds_list, lf)
                print()

        md_path = RESULTS_DIR / f'{cfg["family"]}_{cfg["size"]}.md'
        md_path.write_text(
            f'# {cfg["label"]} posthoc ablation\n\n```\n{_renderTerminal(buf.getvalue())}\n```\n'
        )
        print(f'[saved] {md_path}')


if __name__ == '__main__':
    main()
