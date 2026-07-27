"""Post-hoc benchmark: all refinement methods on a common dataset subset.

Conditions (Normal)
-------------------
raw          : raw flow samples
is           : global IS with PSIS
imhMarginal  : IMH mode='marginal' (Normal, Rao-Blackwellised) or 'global' (other). Note:
               'global' still degrades calibration vs raw/is on non-Normal likelihoods (it
               never MH-corrects rfx — see metabeta/posthoc/metropolis.py's TODO); 'joint'
               was tried too and was worse (acceptance collapses with many groups). Neither
               is a real fix — see that module's TODO for the actual proposed correction.
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
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.warmnuts import _stackProposals
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.results import Proposal, concatProposalsBatch
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.padding import unpad
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
    p.add_argument('--n-datasets', type=int, default=None, help='cap on datasets per model (default: use the entire split)')
    p.add_argument('--include-svgd', action='store_true', help='also run the (slow) SVGD condition')
    p.add_argument('--include-warmnuts', action='store_true', help='also run the (slow) warm-started NUTS condition')
    return p.parse_args()
# fmt: on


def buildModels(families: list[str], sizes: list[str]) -> list[dict]:
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
                    ckpt=_ckpt_dir(family, size, seed) / 'best.pt',
                    data_dir=DATA_DIR / f'{size}-{FAMILY_INITIAL[family]}-sampled',
                    likelihood_family=LIKELIHOOD_FAMILY[family],
                )
            )
    return models


BATCH_SIZE = 4   # sub-batch size for torch-based methods

# Flow samples for torch-based methods (svgd / imh / is / raw).
# IMH requires exactly N_CHAINS × N_STEPS samples (imported as IMH_N_SAMPLES).
N_SAMPLES = 500


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
    col = Collection(path, permute=False)
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
) -> tuple[list, list]:
    """Draw rescaled flow proposals in sub-batches; return (proposals, batches)."""
    proposals, batches = [], []
    with torch.no_grad():
        for i in range(0, len(items), BATCH_SIZE):
            batch = collateGrouped(items[i : i + BATCH_SIZE])
            proposal = model.estimate(batch, n_samples=n_samples)
            proposal.rescale(batch['sd_y'])
            batch = rescaleData(batch)
            proposals.append(proposal)
            batches.append(batch)
    return proposals, batches


def runRaw(proposals, full_batch, lf):
    proposal = concatProposalsBatch(proposals)
    print(summaryTable(getSummary(proposal, full_batch, likelihood_family=lf), lf))


def runIS(proposals, batches, full_batch, lf):
    out = []
    with torch.no_grad():
        for p, batch in zip(proposals, batches):
            sampler = ImportanceSampler(
                batch,
                full=False,
                corr_prior=True,
                pareto=True,
                likelihood_family=lf,
            )
            out.append(sampler(p))
    proposal = concatProposalsBatch(out)
    print(summaryTable(getSummary(proposal, full_batch, likelihood_family=lf), lf))


def runNutsFromNpz(npz_path: Path, ds_list: list, tensor_batch: dict, full_batch: dict, lf: int):
    """Extract pre-computed NUTS results from test.fit.npz and evaluate."""
    data = np.load(npz_path, allow_pickle=True)
    n_ds = len(ds_list)
    has_se = hasSigmaEps(lf)

    proposals = []
    total_divs = 0
    total_time = 0.0

    for i, ds in enumerate(ds_list):
        d_i = int(ds['d'])
        q_i = int(ds['q'])
        m_i = int(ds['m'])
        n_s = data['nuts_ffx'].shape[-1]

        ffx = torch.as_tensor(data['nuts_ffx'][i, :d_i, :]).float().T          # (n_s, d_i)
        sigma_rfx = torch.as_tensor(data['nuts_sigma_rfx'][i, :q_i, :]).float().T  # (n_s, q_i)
        parts = [ffx, sigma_rfx]
        if has_se:
            sigma_eps = torch.as_tensor(data['nuts_sigma_eps'][i, 0, :]).float()
            parts.append(sigma_eps.unsqueeze(-1))                               # (n_s, 1)
        samples_g = torch.cat(parts, dim=-1).unsqueeze(0)                      # (1, n_s, D)

        rfx_raw = torch.as_tensor(data['nuts_rfx'][i, :q_i, :m_i, :]).float()  # (q_i, m_i, n_s)
        samples_l = rfx_raw.permute(1, 2, 0).unsqueeze(0)                      # (1, m_i, n_s, q_i)

        corr_rfx = torch.as_tensor(
            data['nuts_corr_rfx'][i, :, :, :q_i, :q_i]
        ).float()                                                                # (1, n_s, q_i, q_i)

        proposed = {
            'global': {'samples': samples_g, 'log_prob': torch.zeros(1, n_s)},
            'local': {'samples': samples_l, 'log_prob': torch.zeros(1, m_i, n_s)},
        }
        proposals.append(Proposal(proposed, has_sigma_eps=has_se, corr_rfx=corr_rfx))
        total_divs += int(data['nuts_divergences'][i].sum())
        total_time += float(data['nuts_duration'][i])

    n_s = data['nuts_ffx'].shape[-1]
    reff = float(data['nuts_ess'][:n_ds].mean() / n_s)
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
    models = buildModels(args.families, args.sizes)

    conditions = ['raw', 'is', 'imhMarginal']
    if args.include_svgd:
        conditions.append('svgd')
    if args.split == 'test':
        conditions.append('coldNuts')
    if args.include_warmnuts:
        conditions.append('warmNuts')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in models:
        lf = cfg['likelihood_family']
        buf = io.StringIO()
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            model, epoch = loadModel(cfg['ckpt'])
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
                f'Datasets: {n_ds}  |  n_samples (flow-based): {N_SAMPLES}  |  data: {data_path.name}'
            )
            print('(per-method settings: see individual eval_*.py benchmarks)\n')

            proposals, batches = collectProposals(model, items, N_SAMPLES)
            # IMH requires exactly N_CHAINS × N_STEPS samples — draw a dedicated set.
            imh_proposals, imh_batches = collectProposals(model, items, IMH_N_SAMPLES)

            for cond in conditions:
                print('=' * 65)
                print(f'  {cond}')
                print('=' * 65)
                if cond == 'raw':
                    runRaw(proposals, full_batch, lf)
                elif cond == 'is':
                    runIS(proposals, batches, full_batch, lf)
                elif cond == 'imhMarginal':
                    imh_mode = 'marginal' if lf == 0 else 'global'
                    runIMH(imh_mode, imh_proposals, imh_batches, full_batch, lf)
                elif cond == 'svgd':
                    runSVGD(proposals, batches, full_batch, lf)
                elif cond == 'coldNuts':
                    runNutsFromNpz(fit_npz, ds_list, tensor_batch, full_batch, lf)
                elif cond == 'warmNuts':
                    runWarmNuts(model, tensor_batch, ds_list, lf)
                print()

        md_path = RESULTS_DIR / f'{cfg["family"]}_{cfg["size"]}.md'
        md_path.write_text(
            f'# {cfg["label"]} posthoc ablation\n\n```\n{_renderTerminal(buf.getvalue())}\n```\n'
        )
        print(f'[saved] {md_path}')


if __name__ == '__main__':
    main()
