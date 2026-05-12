"""Pre-compute and cache analytical fit summaries (NUTS / ADVI).

Run once after fitting is complete, before submitting training jobs.
All training runs on the same validation dataset will then load the cache
immediately instead of recomputing it (and racing to write it) on startup.

Typical layout:
    valid.fit.npz — NUTS only
    test.fit.npz  — NUTS + ADVI

Methods not present in a fit file are silently skipped.

Usage (from repo root):
    uv run python metabeta/evaluation/cache_nuts_summary.py --data_id small-n-sampled --partition valid
    uv run python metabeta/evaluation/cache_nuts_summary.py --data_id small-n-sampled --partition test
"""

import argparse
import logging
from pathlib import Path

import torch

from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.evaluation import EvaluationSummary, Proposal
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import corrToLower
from metabeta.evaluation.summary import getSummary, summaryTable

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DIR = Path(__file__).resolve().parent


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


def _cache(
    batch: dict,
    method: str,
    cache_path: Path,
    likelihood_family: int,
    rescale: bool,
    d_corr: int,
    force: bool,
    fit_path: Path,
) -> None:
    if f'{method}_ffx' not in batch:
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

    proposal = _buildProposal(batch, method, d_corr)

    batch_for_summary = batch
    if rescale:
        proposal.rescale(batch['sd_y'])
        batch_for_summary = rescaleData(batch)

    logger.info('Computing %s summary...', method)
    summary = getSummary(proposal, batch_for_summary, likelihood_family=likelihood_family)
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
    dl = Dataloader(fit_path, batch_size=8, sortish=True, max_d=max_d, max_q=max_q)
    batch = dl.fullBatch()
    del dl

    for method in ('nuts', 'advi'):
        cache_path = data_dir / f'summary_{args.partition}_{method}.pt'
        _cache(batch, method, cache_path, likelihood_family, rescale, d_corr, args.force, fit_path)


if __name__ == '__main__':
    main()
