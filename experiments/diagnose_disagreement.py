"""
experiments/diagnose_disagreement.py

Per-dataset MB vs NUTS disagreement analysis on the full test set (no convergence pre-filter).

Outputs:
  1. Top-N most disagreeing datasets with NUTS diagnostics and dataset properties
  2. Spearman correlations: disagreement ~ NUTS quality + dataset properties
     (full set and NUTS-converged subset separately)
  3. Summary binned by div_rate quartile and by d / q

Usage (from metabeta/experiments/):
  uv run python diagnose_disagreement.py \\
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

logger = logging.getLogger('diagnose_disagreement')


# ---------------------------------------------------------------------------
# Setup / loading
# ---------------------------------------------------------------------------


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Per-dataset MB vs NUTS disagreement analysis',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--prefix',        type=str, default='best')
    parser.add_argument('--data_id_valid', type=str)
    parser.add_argument('--n_samples',     type=int, default=512)
    parser.add_argument('--batch_size',    type=int, default=8)
    parser.add_argument('--top_n',         type=int, default=20)
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

    for k in ('n_samples', 'batch_size', 'device', 'verbosity', 'seed', 'data_id_valid', 'top_n'):
        if hasattr(args, k):
            cfg_dict[k] = getattr(args, k)

    return argparse.Namespace(**cfg_dict)


def _dataPath(cfg, exp_dir: Path) -> Path:
    data_cfg_train = loadDataConfig(cfg.data_id)
    assimilateConfig(cfg, data_cfg_train)
    data_id = loadDataConfig(cfg.data_id_valid)['data_id']
    path = (exp_dir / '..' / 'metabeta' / 'outputs' / 'data' / data_id / datasetFilename('test')).resolve()
    return path.with_suffix('.fit.npz')


def _loadTestData(path: Path, batch_size: int) -> Dataloader:
    # sortish=False to preserve npz order (needed to align with raw npz fields)
    return Dataloader(path, batch_size=batch_size, sortish=False)


def _loadModel(cfg, exp_dir: Path, device) -> Approximator:
    model_cfg_path = (exp_dir / '..' / 'metabeta' / 'configs' / 'models' / f'{cfg.model_id}.yaml').resolve()
    model_cfg = modelFromYaml(
        model_cfg_path, d_ffx=cfg.max_d, d_rfx=cfg.max_q,
        likelihood_family=cfg.likelihood_family,
    )
    model = Approximator(model_cfg).to(device)
    model.eval()
    ckpt_path = Path(cfg._checkpoint_dir) / f'{cfg._checkpoint_prefix}.pt'
    assert ckpt_path.exists(), f'checkpoint not found: {ckpt_path}'
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
    logger.info('Loaded checkpoint: %s', ckpt_path)
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


def _nutsProposal(batch: dict, rescale: bool) -> Proposal:
    ffx, sigma_rfx = batch['nuts_ffx'], batch['nuts_sigma_rfx']
    has_se = 'nuts_sigma_eps' in batch
    parts_g = [ffx, sigma_rfx] + ([batch['nuts_sigma_eps'].unsqueeze(-1)] if has_se else [])
    global_samples = torch.cat(parts_g, dim=-1)
    rfx = batch['nuts_rfx']
    proposed = {
        'global': {'samples': global_samples, 'log_prob': torch.zeros(global_samples.shape[:2])},
        'local':  {'samples': rfx,            'log_prob': torch.zeros(rfx.shape[:-1])},
    }
    p = Proposal(proposed, has_sigma_eps=has_se, corr_rfx=batch.get('nuts_corr_rfx'))
    if rescale:
        p.rescale(batch['sd_y'])
    return p


# ---------------------------------------------------------------------------
# Per-dataset disagreement metrics
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return float('nan')
    x, y = x[valid], y[valid]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 0 else float('nan')


def perDatasetStats(
    p_mb: Proposal,
    p_nuts: Proposal,
    batch: dict,
    d_arr: np.ndarray,
    q_arr: np.ndarray,
    r2_arr: np.ndarray,
) -> list[dict]:
    B = p_mb.ffx.shape[0]
    m_arr = batch['m'].numpy()
    n_arr = batch['n'].numpy()

    # NUTS quality
    div_np = batch['nuts_divergences'].numpy()        # (B, chains)
    n_draws = int(batch['nuts_draws'].item()) if 'nuts_draws' in batch else 1000
    div_rate = div_np.sum(-1) / (div_np.shape[-1] * n_draws)

    def _param_stat(key, fn):
        a = batch[key].numpy().copy().astype(float)
        a[a <= 0] = np.nan
        return fn(a, axis=-1)

    max_rhat = _param_stat('nuts_rhat', np.nanmax)
    min_ess  = _param_stat('nuts_ess',  np.nanmin)
    td_sat   = batch['nuts_max_treedepth'].numpy().mean(-1)

    conv_mask = nutsConvergeMask(batch)

    rows = []
    for b in range(B):
        d_b = int(d_arr[b])
        q_b = int(q_arr[b])
        m_b = int(m_arr[b])

        # Global params: (S, d_b+q_b+1)
        g_mb = torch.cat([
            p_mb.ffx[b, :, :d_b],
            p_mb.sigma_rfx[b, :, :q_b],
            p_mb.sigma_eps[b].unsqueeze(-1),
        ], dim=-1)
        g_nuts = torch.cat([
            p_nuts.ffx[b, :, :d_b],
            p_nuts.sigma_rfx[b, :, :q_b],
            p_nuts.sigma_eps[b].unsqueeze(-1),
        ], dim=-1)

        mu_mb    = g_mb.mean(0).numpy()
        mu_nuts  = g_nuts.mean(0).numpy()
        std_mb   = g_mb.std(0).numpy().clip(1e-8)
        std_nuts = g_nuts.std(0).numpy().clip(1e-8)

        mean_diff_g = float(np.mean(np.abs(mu_mb - mu_nuts) / std_nuts))
        med_lw_g    = float(np.median(np.log(std_mb / std_nuts)))   # log width ratio

        # RFX: (m_b, S, q_b)
        rfx_mb   = p_mb.rfx[b, :m_b, :, :q_b].numpy()
        rfx_nuts = p_nuts.rfx[b, :m_b, :, :q_b].numpy()
        mu_rfx_mb    = rfx_mb.mean(1).reshape(-1)
        mu_rfx_nuts  = rfx_nuts.mean(1).reshape(-1)
        std_rfx_nuts = rfx_nuts.std(1).reshape(-1).clip(1e-8)
        std_rfx_mb   = rfx_mb.std(1).reshape(-1).clip(1e-8)

        mean_diff_rfx = float(np.mean(np.abs(mu_rfx_mb - mu_rfx_nuts) / std_rfx_nuts))
        med_lw_rfx    = float(np.median(np.log(std_rfx_mb / std_rfx_nuts)))

        rows.append({
            'idx':           b,
            'mean_diff_g':   mean_diff_g,
            'med_lw_g':      med_lw_g,
            'mean_diff_rfx': mean_diff_rfx,
            'med_lw_rfx':    med_lw_rfx,
            'div_rate':      float(div_rate[b]),
            'max_rhat':      float(max_rhat[b]),
            'min_ess':       float(min_ess[b]),
            'td_sat':        float(td_sat[b]),
            'converged':     bool(conv_mask[b]),
            'd':             d_b,
            'q':             q_b,
            'm':             m_b,
            'n':             int(n_arr[b]),
            'r2':            float(r2_arr[b]),
        })

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def printTopDisagreements(rows: list[dict], n: int) -> None:
    top = sorted(rows, key=lambda r: r['mean_diff_g'], reverse=True)[:n]
    headers = ['idx', 'diff_g', 'lw_g', 'diff_rfx', 'div_rate', 'rhat', 'ess', 'td', 'd', 'q', 'm', 'n', 'r2', 'conv']
    disp = []
    for r in top:
        disp.append([
            r['idx'],
            f"{r['mean_diff_g']:.3f}",
            f"{r['med_lw_g']:+.3f}",
            f"{r['mean_diff_rfx']:.3f}",
            f"{r['div_rate']:.4f}",
            f"{r['max_rhat']:.3f}",
            f"{r['min_ess']:.0f}",
            f"{r['td_sat']:.3f}",
            r['d'], r['q'], r['m'], r['n'],
            f"{r['r2']:.2f}",
            'Y' if r['converged'] else 'N',
        ])
    print(f'\n=== Top {n} datasets by global disagreement (mean |μ_MB−μ_NUTS|/σ_NUTS) ===')
    print(tabulate(disp, headers=headers, tablefmt='simple'))
    print('  lw_g = median log(σ_MB/σ_NUTS); positive = MB wider\n')


def printCorrelations(rows: list[dict]) -> None:
    predictors = [
        ('div_rate',  'NUTS divergence rate'),
        ('max_rhat',  'NUTS max R-hat'),
        ('min_ess',   'NUTS min ESS (neg assoc expected)'),
        ('td_sat',    'NUTS treedepth saturation'),
        ('d',         'd (fixed effects)'),
        ('q',         'q (rfx dims)'),
        ('m',         'm (groups)'),
        ('n',         'n (total obs)'),
        ('r2',        'R² (simulated)'),
    ]

    def _table(subset, label):
        y = np.array([r['mean_diff_g'] for r in subset])
        tbl = []
        for key, desc in predictors:
            x = np.array([r[key] for r in subset])
            tbl.append({'Predictor': key, 'Description': desc, 'Spearman r': f'{_spearman(x, y):+.3f}'})
        print(f'\n=== Spearman r with mean_diff_g — {label} (n={len(subset)}) ===')
        print(tabulate(tbl, headers='keys', tablefmt='simple'))

    _table(rows, 'all datasets')
    _table([r for r in rows if r['converged']], 'NUTS-converged subset')
    print()


def printBinnedSummary(rows: list[dict]) -> None:
    def _med(vals):
        v = [x for x in vals if np.isfinite(x)]
        return f'{np.median(v):.4f}' if v else 'nan'

    # By NUTS convergence
    conv   = [r['mean_diff_g'] for r in rows if r['converged']]
    noconv = [r['mean_diff_g'] for r in rows if not r['converged']]
    print('=== Disagreement by NUTS convergence ===')
    print(tabulate([
        {'Group': f'converged     (n={len(conv)})',   'median diff_g': _med(conv)},
        {'Group': f'not converged (n={len(noconv)})', 'median diff_g': _med(noconv)},
    ], headers='keys', tablefmt='simple'))
    print()

    # By div_rate quartile
    all_dr = np.array([r['div_rate'] for r in rows])
    thresholds = np.quantile(all_dr, [0, 0.25, 0.5, 0.75, 1.0])
    bin_rows = []
    for i in range(4):
        lo, hi = thresholds[i], thresholds[i + 1]
        subset = [r['mean_diff_g'] for r in rows if lo <= r['div_rate'] <= hi]
        bin_rows.append({'div_rate bin': f'[{lo:.4f}, {hi:.4f}]', 'n': len(subset), 'median diff_g': _med(subset)})
    print('=== Disagreement by div_rate quartile ===')
    print(tabulate(bin_rows, headers='keys', tablefmt='simple'))
    print()

    # By d and q
    for dim, label in [('d', 'fixed effects'), ('q', 'rfx dims')]:
        vals = sorted(set(r[dim] for r in rows))
        tbl = [{'d' if dim == 'd' else 'q': v,
                'n': sum(1 for r in rows if r[dim] == v),
                'median diff_g': _med([r['mean_diff_g'] for r in rows if r[dim] == v])}
               for v in vals]
        print(f'=== Disagreement by {dim} ({label}) ===')
        print(tabulate(tbl, headers='keys', tablefmt='simple'))
        print()


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
    dl        = _loadTestData(data_path, cfg.batch_size)
    model     = _loadModel(cfg, exp_dir, device)

    # Extra fields not loaded by Dataloader — read directly in npz order (sortish=False)
    raw = np.load(data_path, allow_pickle=True)
    d_arr  = raw['d'].astype(int)
    q_arr  = raw['q'].astype(int)
    r2_arr = raw['r_squared'].astype(float) if 'r_squared' in raw else np.full(len(d_arr), np.nan)

    full_batch    = dl.fullBatch()
    proposal_mb   = _sampleMB(model, dl, cfg.n_samples, device, cfg.rescale)
    proposal_nuts = _nutsProposal(full_batch, cfg.rescale)

    rows = perDatasetStats(proposal_mb, proposal_nuts, full_batch, d_arr, q_arr, r2_arr)

    printTopDisagreements(rows, cfg.top_n)
    printCorrelations(rows)
    printBinnedSummary(rows)


if __name__ == '__main__':
    main()
