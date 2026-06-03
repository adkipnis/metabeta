"""Pre-compute and cache analytical fit summaries (NUTS / ADVI).

Run once after fitting is complete, before submitting training jobs.
All training runs on the same validation dataset will then load the cache
immediately instead of recomputing it (and racing to write it) on startup.

Typical layout:
    valid.fit.npz — NUTS only
    test.fit.npz  — NUTS + ADVI

Methods not present in a fit file are silently skipped.

Usage (from repo root):
    uv run python metabeta/evaluation/cache.py --data_id small-n-sampled --partition valid
    uv run python metabeta/evaluation/cache.py --data_id small-n-sampled --partition test
"""

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.evaluation import AggregatedMetrics, EvaluationSummary, PerDatasetMetrics, Proposal
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import corrToLower
from metabeta.evaluation.point import getCorrelation, getRMSE
from metabeta.evaluation.intervals import getCoverageErrors
from metabeta.evaluation.summary import (
    _averageOverAlpha,
    _rfxJointRanks,
    _rfxRanksToCalibration,
    getSummary,
    summaryTable,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DIR = Path(__file__).resolve().parent

# Datasets per mini-batch. Sortish sampling keeps max(m) and max(n_i) tight
# within each batch, so NUTS rfx tensors stay in the (batch, m_batch, s, q)
# regime rather than exploding to (all_datasets, m_global_max, s, q).
_BATCH_SIZE = 16


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Cache analytical fit summaries.')
    parser.add_argument('--data_id', type=str, required=True,
                        help='Dataset ID (e.g. small-n-sampled)')
    parser.add_argument('--partition', choices=['valid', 'test'], required=True,
                        help='Partition to cache: valid or test')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if a valid cache already exists')
    return parser.parse_args()
# fmt: on


def _buildProposal(
    batch: dict,
    method: str,
    d_corr: int,
) -> Proposal:
    samples_g = [batch[f'{method}_ffx'], batch[f'{method}_sigma_rfx']]
    has_sigma_eps = f'{method}_sigma_eps' in batch
    if has_sigma_eps:
        samples_g.append(batch[f'{method}_sigma_eps'].unsqueeze(-1))
    corr = batch.get(f'{method}_corr_rfx')
    if d_corr > 0 and corr is not None:
        samples_g.append(corrToLower(corr))
    elif d_corr > 0:
        logger.warning('%s: corr_rfx missing, padding zeros for d_corr=%d', method, d_corr)
        samples_g.append(samples_g[0].new_zeros(*samples_g[0].shape[:-1], d_corr))
    return Proposal(
        {
            'global': {'samples': torch.cat(samples_g, dim=-1)},
            'local': {'samples': batch[f'{method}_rfx']},
        },
        has_sigma_eps=has_sigma_eps,
        d_corr=d_corr,
    )


def _smallData(data: dict) -> dict:
    """Extract only the parameter + mask tensors needed for corr/nrmse (no X/y/Z)."""
    keep = ('ffx', 'rfx', 'sigma_rfx', 'sigma_eps', 'corr_rfx',
            'mask_d', 'mask_q', 'mask_m', 'mask_mq', 'mask_corr', 'ns')
    return {k: data[k] for k in keep if k in data}


def _padM(t: torch.Tensor, target_m: int) -> torch.Tensor:
    """Pad the m-dimension of t to target_m.

    Convention: m is the last dim for 2-D tensors (b, m) and the second-to-last
    for 3-D tensors (b, m, q).
    """
    m_dim = -1 if t.dim() == 2 else -2
    pad = target_m - t.shape[m_dim]
    if pad == 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[m_dim] = pad
    return torch.cat([t, t.new_zeros(pad_shape)], dim=m_dim)


def _catSmallData(small_data_list: list[dict]) -> dict:
    """Concatenate per-batch small-data dicts, padding m-dim tensors to global max_m."""
    m_dim_keys = {'rfx', 'mask_m', 'mask_mq', 'ns'}
    global_m = max(sd['rfx'].shape[-2] for sd in small_data_list if 'rfx' in sd)
    out = {}
    for key in small_data_list[0]:
        parts = [sd[key] for sd in small_data_list]
        if key in m_dim_keys:
            parts = [_padM(p, global_m) for p in parts]
        out[key] = torch.cat(parts, dim=0)
    return out


def _catEstimates(estimates_list: list[dict]) -> dict:
    """Concatenate per-batch estimates dicts, padding rfx to global max_m."""
    global_m = max(e['rfx'].shape[-2] for e in estimates_list if 'rfx' in e)
    out = {}
    for key in estimates_list[0]:
        parts = [e[key] for e in estimates_list]
        if key == 'rfx':
            parts = [_padM(p, global_m) for p in parts]
        out[key] = torch.cat(parts, dim=0)
    return out


def _activeCount(small_data: dict, key: str, batch_size: int) -> torch.Tensor:
    """Per-dimension active count used by getAtomicCoverage for the given parameter key."""
    if key == 'ffx':
        return small_data['mask_d'].float().sum(0)          # (d,)
    elif key == 'sigma_rfx':
        return small_data['mask_q'].float().sum(0)          # (q,)
    elif key == 'rfx':
        return small_data['mask_mq'].float().sum((0, 1))    # (q,)  group_weighted
    elif key == 'corr_rfx':
        return small_data['mask_corr'].float().sum(0)       # (d_corr,)
    else:
        return torch.tensor([float(batch_size)])            # sigma_eps: all active


def _mergeCoverage(
    coverage_list: list[dict],
    small_data_list: list[dict],
    batch_sizes: list[int],
) -> dict:
    """Merge per-batch coverage dicts, weighting by per-dimension active count."""
    alphas = list(coverage_list[0].keys())
    out: dict = {}
    for alpha in alphas:
        out[alpha] = {}
        for key in coverage_list[0][alpha]:
            nums, dens = [], []
            for cov_d, sd, b in zip(coverage_list, small_data_list, batch_sizes):
                cov = cov_d[alpha][key]
                n = _activeCount(sd, key, b)
                nums.append(n * torch.nan_to_num(cov, nan=0.0))
                dens.append(n)
            total_n = sum(dens)
            total_num = sum(nums)
            out[alpha][key] = torch.where(
                total_n > 0,
                total_num / total_n.clamp_min(1.0),
                torch.zeros_like(total_num),
            )
    return out


def _mergeSummaries(
    partials: list[EvaluationSummary],
    small_data_list: list[dict],
    batch_sizes: list[int],
    likelihood_family: int,
    all_rfx_ranks: list[float],
) -> EvaluationSummary:
    total = sum(batch_sizes)

    # --- per-dataset fields: concatenate ---
    def _cat1(fn):
        parts = [fn(p.per_dataset) for p in partials]
        return None if parts[0] is None else torch.cat(parts)

    def _cat2(fn):
        parts = [fn(p.per_dataset) for p in partials]
        return None if parts[0] is None else torch.cat(parts, dim=-1)

    per_dataset = PerDatasetMetrics(
        posterior_nll=torch.cat([p.per_dataset.posterior_nll for p in partials]),
        loo_nll=_cat1(lambda p: p.loo_nll),
        loo_pareto_k=_cat1(lambda p: p.loo_pareto_k),
        pp_fit=_cat1(lambda p: p.pp_fit),
        pp_cov_coverage=_cat2(lambda p: p.pp_cov_coverage),
        pp_cov_width=_cat2(lambda p: p.pp_cov_width),
        sample_efficiency=_cat1(lambda p: p.sample_efficiency),
        pareto_k=_cat1(lambda p: p.pareto_k),
        prior_nll=_cat1(lambda p: p.prior_nll),
    )

    # --- aggregated: recompute corr/nrmse from pooled data; average the rest ---
    all_estimates = _catEstimates([p.aggregated.estimates for p in partials])
    all_small = _catSmallData(small_data_list)
    corr = getCorrelation(all_estimates, all_small)
    nrmse = getRMSE(all_estimates, all_small, normalize=True)

    coverage = _mergeCoverage(
        [p.aggregated.coverage for p in partials], small_data_list, batch_sizes
    )
    coverage_error = getCoverageErrors(coverage, log_ratio=False)
    log_coverage_ratio = getCoverageErrors(coverage, log_ratio=True)

    rfx_joint_ece, rfx_joint_eace = (
        _rfxRanksToCalibration(all_rfx_ranks) if all_rfx_ranks else (None, None)
    )

    aggregated = AggregatedMetrics(
        corr=corr,
        nrmse=nrmse,
        coverage=coverage,
        ece=_averageOverAlpha(coverage_error),
        eace=_averageOverAlpha(coverage_error, absolute=True),
        lcr=_averageOverAlpha(log_coverage_ratio),
        abs_lcr=_averageOverAlpha(log_coverage_ratio, absolute=True),
        estimates=all_estimates,
        rfx_joint_ece=rfx_joint_ece,
        rfx_joint_eace=rfx_joint_eace,
    )

    tpd_pairs = [(batch_sizes[i], p.tpd) for i, p in enumerate(partials) if p.tpd is not None]
    tpd_total = sum(w for w, _ in tpd_pairs)
    tpd = sum(w * v for w, v in tpd_pairs) / tpd_total if tpd_pairs else None

    return EvaluationSummary(per_dataset=per_dataset, aggregated=aggregated, tpd=tpd)


def _cache(
    dl: Dataloader,
    method: str,
    cache_path: Path,
    likelihood_family: int,
    rescale: bool,
    d_corr: int,
    force: bool,
    fit_path: Path,
) -> None:
    if f'{method}_ffx' not in dl.dataset.raw:
        logger.info('%s: no samples in %s, skipping.', method, fit_path.name)
        return

    if not force and cache_path.exists():
        if cache_path.stat().st_mtime >= fit_path.stat().st_mtime:
            logger.info('Cache is up to date: %s', cache_path)
            try:
                summary = EvaluationSummary.load(cache_path)
                print(f'\n{method} ({fit_path.stem}):\n{summaryTable(summary, likelihood_family)}')
                return
            except Exception as e:
                logger.warning('Cache unreadable (%s), recomputing.', e)

    n_total = len(dl.dataset)
    partials: list[EvaluationSummary] = []
    small_data_list: list[dict] = []
    batch_sizes: list[int] = []
    all_rfx_ranks: list[float] = []

    with tqdm(total=n_total, desc=method, unit='ds') as pbar:
        for batch in dl:
            b = batch['y'].shape[0]

            proposal = _buildProposal(batch, method, d_corr)
            batch_for_summary = batch
            if rescale:
                proposal.rescale(batch['sd_y'])
                batch_for_summary = rescaleData(batch)

            partial = getSummary(proposal, batch_for_summary, likelihood_family=likelihood_family)
            all_rfx_ranks.extend(_rfxJointRanks(proposal, batch_for_summary))
            partials.append(partial)
            small_data_list.append(_smallData(batch_for_summary))
            batch_sizes.append(b)

            del batch, proposal, batch_for_summary, partial
            pbar.update(b)

    print(f'  [{method}] merging ...')
    summary = _mergeSummaries(partials, small_data_list, batch_sizes, likelihood_family, all_rfx_ranks)
    summary.save(cache_path)
    logger.info('Saved: %s', cache_path)
    print(f'\n{method} ({fit_path.stem}):\n{summaryTable(summary, likelihood_family)}')


def main() -> None:
    args = setup()
    data_cfg = loadDataConfig(args.data_id)

    data_dir = Path(DIR, '..', 'outputs', 'data', args.data_id)
    max_d = data_cfg['max_d']
    max_q = data_cfg['max_q']
    likelihood_family = data_cfg['likelihood_family']
    rescale = likelihood_family == 0
    d_corr = max_q * (max_q - 1) // 2 if max_q >= 2 else 0

    fit_path = data_dir / f'{args.partition}.fit.npz'
    if not fit_path.exists():
        logger.error('Fit file not found: %s', fit_path)
        return

    logger.info('Loading %s', fit_path)
    dl = Dataloader(fit_path, batch_size=_BATCH_SIZE, sortish=True, max_d=max_d, max_q=max_q)
    logger.info('Datasets: %d', len(dl.dataset))

    for method in ('nuts', 'advi'):
        cache_path = data_dir / f'summary_{args.partition}_{method}.pt'
        _cache(dl, method, cache_path, likelihood_family, rescale, d_corr, args.force, fit_path)


if __name__ == '__main__':
    main()
