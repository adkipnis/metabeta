"""Post-hoc benchmark: all refinement methods on a common dataset subset.

Conditions (Normal)
-------------------
raw          : raw flow samples
is           : global IS with PSIS
laplace      : MAP (Adam, NCP) + Laplace Gaussian approximation
laplaceIS    : laplace → full IS correction (PSIS)
imhMarginal  : IMH mode='marginal' (Normal) or 'joint' (other) — Rao-Blackwellised where possible
cd           : coordinate descent — marginal MAP θ_g + Gibbs rfx sample
svgd         : SVGD with per-dim bandwidth + cosine LR decay
coldNuts     : NUTS results extracted from test.fit.npz (pre-computed, no rerun)
warmNuts     : warm-started NUTS (flow samples initialise PyMC chains)

All methods are evaluated on the same N_DATASETS datasets.
Run functions are imported from the individual eval scripts; see those files
for per-method settings (e.g. IMH chains/steps, NUTS tune/draws).

Data loading uses Collection + collateGrouped so that per-dataset unpadded
dicts are available for NUTS-based methods.

Run from repo root:
    uv run python benchmarks/benchmark_posthoc.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Make sibling eval scripts importable without installing benchmarks as a package.
sys.path.insert(0, str(Path(__file__).parent))

from eval_coordinate import runCD                                      # noqa: E402
from eval_imh import runIMH, N_SAMPLES as IMH_N_SAMPLES               # noqa: E402
from eval_laplace import runLaplace, runLaplaceIS                      # noqa: E402
from eval_svgd import runSVGD                                          # noqa: E402
from eval_warmnuts import runWarmNuts                                  # noqa: E402

from metabeta.evaluation.summary import getSummary, summaryTable
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.warmnuts import _stackProposals
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.evaluation import Proposal, concatProposalsBatch
from metabeta.utils.families import hasSigmaEps
from metabeta.utils.padding import unpad
from metabeta.utils.preprocessing import rescaleData

# ---------------------------------------------------------------------------
# Dataset limit — all methods evaluated on the same N_DATASETS datasets.
# ---------------------------------------------------------------------------
N_DATASETS = 128
BATCH_SIZE = 4   # sub-batch size for torch-based methods

# Flow samples for torch-based methods (laplace / cd / svgd / imh / is / raw).
# IMH requires exactly N_CHAINS × N_STEPS samples (imported as IMH_N_SAMPLES).
N_SAMPLES = 500

# ---------------------------------------------------------------------------
# Per-model config
# ---------------------------------------------------------------------------
MODELS = [
    dict(
        label='Normal',
        ckpt=Path('outputs/checkpoints/normal_dsmall-n-mixed_msmall_s42/best.pt'),
        data_dir=Path('outputs/data/small-n-sampled'),
        likelihood_family=0,
    ),
    dict(
        label='Bernoulli',
        ckpt=Path('outputs/checkpoints/bernoulli_dsmall-b-mixed_msmall_s42/best.pt'),
        data_dir=Path('outputs/data/small-b-sampled'),
        likelihood_family=1,
    ),
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def loadModel(ckpt: Path) -> tuple[Approximator, int]:
    payload = torch.load(ckpt, map_location='cpu')
    model_cfg = ApproximatorConfig(**payload['model_cfg'])
    model = Approximator(model_cfg)
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model, payload['epoch']


def loadData(path: Path, n_limit: int) -> tuple[list, list[dict], dict, dict]:
    """Load up to n_limit datasets.

    Returns
    -------
    items        : list of Collection items; used to build sub-batches
    ds_list      : list of fully-unpadded numpy dicts for NUTS / buildPymc
    tensor_batch : single collated tensor dict, un-rescaled (for NUTS flow init)
    full_batch   : rescaled tensor_batch (ground truth for evaluation)
    """
    col = Collection(path, permute=False)
    n = min(n_limit, len(col))
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
                batch, full=False, corr_prior=True, pareto=True, likelihood_family=lf,
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
    print(f'  divergences={total_divs}  reff={reff:.3f}  time/ds={total_time / n_ds:.1f}s  '
          f'(total={total_time:.0f}s)')

    target_d = tensor_batch['ffx'].shape[-1]
    target_q = tensor_batch['sigma_rfx'].shape[-1]
    merged = _stackProposals(proposals, target_d=target_d, target_q=target_q)
    merged.rescale(tensor_batch['sd_y'][:n_ds])
    merged.reff = reff
    print(summaryTable(getSummary(merged, full_batch, likelihood_family=lf), lf))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for cfg in MODELS:
    lf = cfg['likelihood_family']
    model, epoch = loadModel(cfg['ckpt'])
    print(f'\n{"#" * 70}')
    print(f'#  {cfg["label"]}  (epoch={epoch}  params={model.n_params:,}  '
          f'd_ffx={model.d_ffx}  d_rfx={model.d_rfx})')
    print(f'{"#" * 70}')

    fit_npz = cfg['data_dir'] / 'test.fit.npz'
    data_path = fit_npz if fit_npz.exists() else cfg['data_dir'] / 'valid.npz'
    items, ds_list, tensor_batch, full_batch = loadData(data_path, N_DATASETS)
    n_ds = len(ds_list)
    print(f'Datasets: {n_ds}  |  n_samples (flow-based): {N_SAMPLES}  |  data: {data_path.name}')
    print('(per-method settings: see individual eval_*.py benchmarks)\n')

    proposals, batches = collectProposals(model, items, N_SAMPLES)
    # IMH requires exactly N_CHAINS × N_STEPS samples — draw a dedicated set.
    imh_proposals, imh_batches = collectProposals(model, items, IMH_N_SAMPLES)

    laplace_out = None
    conditions = ('raw', 'is', 'laplace', 'laplaceIS', 'imhMarginal', 'cd', 'svgd',
                  'coldNuts', 'warmNuts')
    for cond in conditions:
        if cond == 'coldNuts' and not fit_npz.exists():
            continue
        print('=' * 65)
        print(f'  {cond}')
        print('=' * 65)
        if cond == 'raw':
            runRaw(proposals, full_batch, lf)
        elif cond == 'is':
            runIS(proposals, batches, full_batch, lf)
        elif cond == 'laplace':
            laplace_out = runLaplace(proposals, batches, full_batch, lf)
        elif cond == 'laplaceIS':
            runLaplaceIS(laplace_out, batches, full_batch, lf)
        elif cond == 'imhMarginal':
            imh_mode = 'marginal' if lf == 0 else 'joint'
            runIMH(imh_mode, imh_proposals, imh_batches, full_batch, lf)
        elif cond == 'cd':
            runCD(proposals, batches, full_batch, lf)
        elif cond == 'svgd':
            runSVGD(proposals, batches, full_batch, lf)
        elif cond == 'coldNuts':
            runNutsFromNpz(fit_npz, ds_list, tensor_batch, full_batch, lf)
        elif cond == 'warmNuts':
            runWarmNuts(model, tensor_batch, ds_list, lf)
        print()
