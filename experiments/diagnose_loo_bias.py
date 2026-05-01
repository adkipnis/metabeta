"""
experiments/diagnose_loo_bias.py

Answers: under which NUTS exclusion criteria is the LOO-NLL comparison least biased?

Key bias mechanism: NUTS divergences cause the posterior to be too concentrated
(sampler skips high-curvature tails), making NUTS LOO-NLL artificially low.
Strict exclusion removes these datasets; liberal keeps them. If the MB-NUTS gap
changes significantly between modes, the exclusion choice is load-bearing.

Outputs per exclusion mode (none / liberal / strict):
  - N datasets included
  - Median MB and NUTS LOO-NLL
  - Median and IQR of gap (MB − NUTS)  [positive = MB worse]
  - Pareto-k diagnostics (flagging unreliable LOO estimates)

Also shows gap binned by div_rate quartile (all datasets, no exclusion).

Usage (from metabeta/experiments/):
  uv run python diagnose_loo_bias.py \\
      --checkpoint ../metabeta/outputs/checkpoints/<run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.utils.logger import setupLogging
from metabeta.utils.io import setDevice, datasetFilename
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import modelFromYaml, assimilateConfig, loadDataConfig
from metabeta.utils.templates import loadConfigFromCheckpoint
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch, nutsConvergeMask
from metabeta.models.approximator import Approximator
from metabeta.evaluation.predictive import getPosteriorPredictive, psisLooNLL

logger = logging.getLogger('diagnose_loo_bias')


# ---------------------------------------------------------------------------
# Setup / loading
# ---------------------------------------------------------------------------


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='LOO-NLL bias analysis under different NUTS exclusion criteria',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--prefix',        type=str, default='best')
    parser.add_argument('--data_id_valid', type=str)
    parser.add_argument('--n_samples',     type=int, default=512)
    parser.add_argument('--nuts_subsample',type=int, default=1000,
                        help='Subsample NUTS draws to this count for LOO (saves memory)')
    parser.add_argument('--chunk',         type=int, default=16,
                        help='Datasets per chunk for LOO computation')
    parser.add_argument('--device',        type=str, default='cpu')
    parser.add_argument('--verbosity',     type=int, default=1)
    parser.add_argument('--seed',          type=int, default=42)
    # fmt: on
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg_dict = loadConfigFromCheckpoint(checkpoint_path)
    cfg_dict['_checkpoint_dir'] = str(checkpoint_path)
    cfg_dict['_checkpoint_prefix'] = getattr(args, 'prefix', 'best')
    cfg_dict.setdefault('rescale', True)

    for k in ('n_samples', 'nuts_subsample', 'chunk', 'device', 'verbosity', 'seed', 'data_id_valid'):
        if hasattr(args, k):
            cfg_dict[k] = getattr(args, k)

    return argparse.Namespace(**cfg_dict)


def _dataPath(cfg, exp_dir: Path) -> Path:
    data_cfg_train = loadDataConfig(cfg.data_id)
    assimilateConfig(cfg, data_cfg_train)
    data_id = loadDataConfig(cfg.data_id_valid)['data_id']
    path = (exp_dir / '..' / 'metabeta' / 'outputs' / 'data' / data_id / datasetFilename('test')).resolve()
    return path.with_suffix('.fit.npz')


def _loadModel(cfg, exp_dir: Path, device) -> Approximator:
    model_cfg_path = (exp_dir / '..' / 'metabeta' / 'configs' / 'models' / f'{cfg.model_id}.yaml').resolve()
    model_cfg = modelFromYaml(
        model_cfg_path, d_ffx=cfg.max_d, d_rfx=cfg.max_q,
        likelihood_family=cfg.likelihood_family,
    )
    model = Approximator(model_cfg).to(device)
    model.eval()
    ckpt = Path(cfg._checkpoint_dir) / f'{cfg._checkpoint_prefix}.pt'
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['model_state'])
    logger.info('Loaded checkpoint: %s', ckpt)
    return model


@torch.inference_mode()
def _sampleMB(model, dl: Dataloader, n_samples: int, device, rescale: bool) -> Proposal:
    proposals = []
    for batch in tqdm(dl, desc='  MB sampling'):
        batch = toDevice(batch, device)
        p = model.estimate(batch, n_samples=n_samples)
        if rescale:
            p.rescale(batch['sd_y'])
        p.to('cpu')
        proposals.append(p)
    return concatProposalsBatch(proposals)


def _nutsProposal(batch: dict, rescale: bool, subsample: int | None = None) -> Proposal:
    """Build Proposal from stored NUTS samples, optionally subsampling draws."""
    ffx        = batch['nuts_ffx']        # (B, S, d)
    sigma_rfx  = batch['nuts_sigma_rfx']  # (B, S, q)
    has_se     = 'nuts_sigma_eps' in batch
    sigma_eps  = batch['nuts_sigma_eps'].unsqueeze(-1) if has_se else None  # (B, S, 1)

    if subsample is not None and ffx.shape[1] > subsample:
        idx = torch.randperm(ffx.shape[1])[:subsample]
        ffx       = ffx[:, idx]
        sigma_rfx = sigma_rfx[:, idx]
        if sigma_eps is not None:
            sigma_eps = sigma_eps[:, idx]
        rfx = batch['nuts_rfx'][:, :, idx, :]  # (B, m, S, q)
    else:
        rfx = batch['nuts_rfx']

    parts_g = [ffx, sigma_rfx] + ([sigma_eps] if has_se else [])
    global_samples = torch.cat(parts_g, dim=-1)
    proposed = {
        'global': {'samples': global_samples, 'log_prob': torch.zeros(global_samples.shape[:2])},
        'local':  {'samples': rfx,            'log_prob': torch.zeros(rfx.shape[:-1])},
    }
    p = Proposal(proposed, has_sigma_eps=has_se, corr_rfx=batch.get('nuts_corr_rfx'))
    if rescale:
        p.rescale(batch['sd_y'])
    return p


def _nutsReff(batch: dict, subsample: int | None = None) -> float:
    """Estimate mean reff = mean_ESS / n_draws for PSIS calibration."""
    ess = batch['nuts_ess'].numpy().copy().astype(float)
    ess[ess <= 0] = np.nan
    min_ess = np.nanmin(ess, axis=-1)          # (B,)
    n_draws = int(batch['nuts_draws'].item()) if 'nuts_draws' in batch else 1000
    n_chains = batch['nuts_divergences'].shape[-1]
    denom = float(subsample if subsample is not None else n_chains * n_draws)
    reff = float(np.nanmedian(min_ess) / denom)
    return max(min(reff, 1.0), 0.01)


# ---------------------------------------------------------------------------
# Per-dataset LOO-NLL computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def computeLooNll(
    proposal: Proposal,
    batch: dict,
    likelihood_family: int,
    reff: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSIS-LOO NLL per dataset. Returns (loo_nll, max_k), both (B,)."""
    B = proposal.ffx.shape[0]
    loos, ks = [], []

    for start in tqdm(range(0, B, chunk), desc='  LOO', leave=False):
        end = min(start + chunk, B)
        p_c = proposal.slice_b(start, end)
        d_c = {k: v[start:end] for k, v in batch.items() if torch.is_tensor(v) and v.shape[0] == B}

        pp_c   = getPosteriorPredictive(p_c, d_c, likelihood_family)
        log_p  = pp_c.log_prob(d_c['y'].unsqueeze(-1))   # (c, m, n, s)
        loo_c, k_c = psisLooNLL(pp_c, d_c, w=None, reff=reff, log_p=log_p)

        loos.append(loo_c.numpy())
        ks.append(k_c.numpy())   # (c,) mean pareto_k per dataset

    return np.concatenate(loos), np.concatenate(ks)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _summaryRow(label: str, mb: np.ndarray, nuts: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {'Mode': label, 'N': 0}
    gap = mb[mask] - nuts[mask]
    return {
        'Mode':               label,
        'N':                  int(mask.sum()),
        'MB  LOO (med)':      f'{np.median(mb[mask]):.4f}',
        'NUTS LOO (med)':     f'{np.median(nuts[mask]):.4f}',
        'Gap med (MB−NUTS)':  f'{np.median(gap):+.4f}',
        'Gap IQR':            f'[{np.percentile(gap,25):+.3f}, {np.percentile(gap,75):+.3f}]',
    }


def printModeComparison(
    mb_loo: np.ndarray,
    nuts_loo: np.ndarray,
    mb_k: np.ndarray,
    nuts_k: np.ndarray,
    batch: dict,
    k_thr: float = 0.7,
) -> None:
    all_mask     = np.ones(len(mb_loo), dtype=bool)
    liberal_mask = nutsConvergeMask(batch, mode='liberal')
    strict_mask  = nutsConvergeMask(batch, mode='strict')
    # Filter only on NUTS k: ensures the reference (NUTS LOO) is reliable.
    # MB k is intentionally NOT used as a filter — filtering on MB k would discard
    # datasets where MB's LOO estimate is unreliable, hiding potential MB failures.
    nuts_k_mask  = nuts_k < k_thr

    rows = [
        _summaryRow('all (no filter)',               mb_loo, nuts_loo, all_mask),
        _summaryRow('liberal',                       mb_loo, nuts_loo, liberal_mask),
        _summaryRow('strict',                        mb_loo, nuts_loo, strict_mask),
        _summaryRow(f'strict + NUTS k<{k_thr}',      mb_loo, nuts_loo, strict_mask & nuts_k_mask),
        _summaryRow(f'liberal + NUTS k<{k_thr}',     mb_loo, nuts_loo, liberal_mask & nuts_k_mask),
    ]
    print('\n=== MB vs NUTS LOO-NLL by exclusion mode (positive gap = MB worse) ===')
    print(tabulate(rows, headers='keys', tablefmt='simple'))
    print(f'  k > {k_thr} fraction (all):    MB {np.mean(mb_k > k_thr):.2f}, NUTS {np.mean(nuts_k > k_thr):.2f}')
    print(f'  k > {k_thr} fraction (strict): MB {np.mean(mb_k[strict_mask] > k_thr):.2f}, NUTS {np.mean(nuts_k[strict_mask] > k_thr):.2f}')
    print(f'  (NUTS k filter only — MB k retained to avoid cherry-picking)\n')


def printDivRateBins(
    mb_loo: np.ndarray,
    nuts_loo: np.ndarray,
    batch: dict,
) -> None:
    div_arr  = batch['nuts_divergences'].numpy()
    n_draws  = int(batch['nuts_draws'].item()) if 'nuts_draws' in batch else 1000
    div_rate = div_arr.sum(-1) / (div_arr.shape[-1] * n_draws)
    gap      = mb_loo - nuts_loo

    thresholds = np.quantile(div_rate, [0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for i in range(4):
        lo, hi = thresholds[i], thresholds[i + 1]
        m = (div_rate >= lo) & (div_rate <= hi)
        g = gap[m]
        rows.append({
            'div_rate bin':       f'[{lo:.4f}, {hi:.4f}]',
            'n':                  int(m.sum()),
            'NUTS LOO (med)':     f'{np.median(nuts_loo[m]):.4f}',
            'Gap med (MB−NUTS)':  f'{np.median(g):+.4f}',
        })
    print('=== LOO-NLL gap binned by div_rate (all datasets) ===')
    print(tabulate(rows, headers='keys', tablefmt='simple'))
    print()


def printKDiagnostics(mb_k: np.ndarray, nuts_k: np.ndarray) -> None:
    def _frac_bad(k): return float(np.mean(k > 0.7))
    print('=== Pareto-k diagnostics (fraction of datasets with mean k > 0.7) ===')
    print(f'  MB:   {_frac_bad(mb_k):.3f}')
    print(f'  NUTS: {_frac_bad(nuts_k):.3f}')
    print('  (k > 0.7 → LOO estimate unreliable for that dataset)\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)

    device  = setDevice(cfg.device)
    exp_dir = Path(__file__).resolve().parent

    data_path = _dataPath(cfg, exp_dir)
    # sortish=False: preserve npz order for nutsConvergeMask alignment
    dl        = Dataloader(data_path, batch_size=8, sortish=False)
    model     = _loadModel(cfg, exp_dir, device)

    full_batch = dl.fullBatch()
    rescale    = cfg.rescale

    logger.info('Computing MB samples...')
    proposal_mb = _sampleMB(model, dl, cfg.n_samples, device, rescale)

    logger.info('Building NUTS proposal (subsample=%d)...', cfg.nuts_subsample)
    proposal_nuts = _nutsProposal(full_batch, rescale, subsample=cfg.nuts_subsample)

    reff_nuts = _nutsReff(full_batch, subsample=cfg.nuts_subsample)
    logger.info('NUTS reff (median): %.3f', reff_nuts)

    logger.info('Computing MB LOO-NLL...')
    mb_loo, mb_k = computeLooNll(
        proposal_mb, full_batch, cfg.likelihood_family, reff=1.0, chunk=cfg.chunk
    )

    logger.info('Computing NUTS LOO-NLL...')
    nuts_loo, nuts_k = computeLooNll(
        proposal_nuts, full_batch, cfg.likelihood_family, reff=reff_nuts, chunk=cfg.chunk
    )

    printModeComparison(mb_loo, nuts_loo, mb_k, nuts_k, full_batch)
    printDivRateBins(mb_loo, nuts_loo, full_batch)
    printKDiagnostics(mb_k, nuts_k)


if __name__ == '__main__':
    main()
