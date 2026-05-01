"""
experiments/posterior_shape.py — Posterior shape diagnostics: MB vs NUTS.

Four diagnostics on the NUTS-converged subset of test datasets:

  1. Width    — std(MB) / std(NUTS) per parameter type
  2. Corr     — mean |Corr_MB − Corr_NUTS| for global parameters
  3. LocalUnc — cond_std_Gauss / marginal_std per rfx dim  [family == 0 only]
  4. Rank     — rank of each NUTS sample in the MB marginal (uniform = shapes match)

Usage (from metabeta/experiments/):
  uv run python posterior_shape.py --checkpoint ../outputs/checkpoints/<run>
  uv run python posterior_shape.py --checkpoint ../outputs/checkpoints/<run> --prefix best
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
from metabeta.utils.dataloader import Dataloader, toDevice, subsetBatch
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import (
    Proposal,
    concatProposalsBatch,
    nutsConvergeMask,
    subsetProposal,
)
from metabeta.models.approximator import Approximator

logger = logging.getLogger('posterior_shape')


# ---------------------------------------------------------------------------
# Setup / loading
# ---------------------------------------------------------------------------


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Posterior shape diagnostics: MB vs NUTS',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--prefix',        type=str, default='latest')
    parser.add_argument('--data_id_valid', type=str)
    parser.add_argument('--n_samples',     type=int, default=512)
    parser.add_argument('--batch_size',    type=int, default=8)
    parser.add_argument('--device',        type=str, default='cpu')
    parser.add_argument('--verbosity',     type=int, default=1)
    parser.add_argument('--seed',          type=int, default=42)
    # fmt: on
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg_dict = loadConfigFromCheckpoint(checkpoint_path)
    cfg_dict['_checkpoint_dir'] = str(checkpoint_path)
    cfg_dict['_checkpoint_prefix'] = getattr(args, 'prefix', 'latest')
    cfg_dict.setdefault('rescale', True)

    for k in ('n_samples', 'batch_size', 'device', 'verbosity', 'seed', 'data_id_valid'):
        if hasattr(args, k):
            cfg_dict[k] = getattr(args, k)

    return argparse.Namespace(**cfg_dict)


def _loadTestData(cfg, dir) -> Dataloader:
    data_cfg_train = loadDataConfig(cfg.data_id)
    assimilateConfig(cfg, data_cfg_train)
    data_id = loadDataConfig(cfg.data_id_valid)['data_id']
    path = (dir / '..' / 'outputs' / 'data' / data_id / datasetFilename('test')).resolve()
    path = path.with_suffix('.fit.npz')
    assert path.exists(), f'data not found: {path}'
    return Dataloader(path, batch_size=cfg.batch_size, sortish=True)


def _loadModel(cfg, dir, device) -> 'Approximator':
    model_cfg_path = (dir / '..' / 'configs' / 'models' / f'{cfg.model_id}.yaml').resolve()
    model_cfg = modelFromYaml(
        model_cfg_path, d_ffx=cfg.max_d, d_rfx=cfg.max_q,
        likelihood_family=cfg.likelihood_family,
    )
    model = Approximator(model_cfg).to(device)
    model.eval()
    path = Path(cfg._checkpoint_dir) / f'{cfg._checkpoint_prefix}.pt'
    assert path.exists(), f'checkpoint not found: {path}'
    model.load_state_dict(torch.load(path, map_location=device)['model_state'])
    logger.info('Loaded checkpoint: %s', path)
    return model


def _batchToProposal(batch: dict, prefix: str, rescale: bool) -> Proposal:
    ffx, sigma_rfx = batch[f'{prefix}_ffx'], batch[f'{prefix}_sigma_rfx']
    has_se = f'{prefix}_sigma_eps' in batch
    parts_g = [ffx, sigma_rfx] + ([batch[f'{prefix}_sigma_eps'].unsqueeze(-1)] if has_se else [])
    global_samples = torch.cat(parts_g, dim=-1)
    rfx = batch[f'{prefix}_rfx']
    proposed = {
        'global': {'samples': global_samples, 'log_prob': torch.zeros(global_samples.shape[:2])},
        'local':  {'samples': rfx,            'log_prob': torch.zeros(rfx.shape[:-1])},
    }
    p = Proposal(proposed, has_sigma_eps=has_se, corr_rfx=batch.get(f'{prefix}_corr_rfx'))
    if rescale:
        p.rescale(batch['sd_y'])
    return p


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


# ---------------------------------------------------------------------------
# Diagnostic 1 — posterior width (std ratios)
# ---------------------------------------------------------------------------


def _widthDiagnostic(p_mb: Proposal, p_nuts: Proposal, batch: dict) -> list[dict]:
    """std(MB) / std(NUTS) per parameter type, median ± IQR across active entries."""
    group_mask = batch['mask_n'].any(-1)

    def _stats(name, mb_v, nuts_v, mask):
        # mb_v, nuts_v: (B, S, d); std over sample dim
        std_mb = mb_v.std(dim=1)
        std_nu = nuts_v.std(dim=1)
        ratio  = std_mb / std_nu.clamp(min=1e-8)
        active = mask if mask is not None else slice(None)
        vals   = ratio[active].flatten().numpy()
        return {
            'Type':      name,
            'std_MB':    float(std_mb[active].mean()),
            'std_NUTS':  float(std_nu[active].mean()),
            'Ratio p50': float(np.median(vals)),
            'Ratio p25': float(np.percentile(vals, 25)),
            'Ratio p75': float(np.percentile(vals, 75)),
        }

    mask_q  = batch.get('mask_q', torch.ones(p_mb.rfx.shape[0], p_mb.rfx.shape[-1], dtype=torch.bool))
    mask_mq = group_mask.unsqueeze(-1) & mask_q.unsqueeze(1)

    # rfx: (B, m, S, q) — std over sample dim=2
    std_mb_l = p_mb.rfx.std(dim=2)
    std_nu_l = p_nuts.rfx.std(dim=2)
    ratio_l  = std_mb_l / std_nu_l.clamp(min=1e-8)
    vals_l   = ratio_l[mask_mq].numpy()

    return [
        _stats('ffx',       p_mb.ffx,                    p_nuts.ffx,                    batch.get('mask_d')),
        _stats('sigma_rfx', p_mb.sigma_rfx,               p_nuts.sigma_rfx,              batch.get('mask_q')),
        _stats('sigma_eps', p_mb.sigma_eps.unsqueeze(-1),  p_nuts.sigma_eps.unsqueeze(-1), None),
        {
            'Type':      'rfx',
            'std_MB':    float(std_mb_l[mask_mq].mean()),
            'std_NUTS':  float(std_nu_l[mask_mq].mean()),
            'Ratio p50': float(np.median(vals_l)),
            'Ratio p25': float(np.percentile(vals_l, 25)),
            'Ratio p75': float(np.percentile(vals_l, 75)),
        },
    ]


# ---------------------------------------------------------------------------
# Diagnostic 2 — global correlation structure
# ---------------------------------------------------------------------------


def _corrDiagnostic(p_mb: Proposal, p_nuts: Proposal, batch: dict) -> dict:
    """Per-dataset mean |Corr_MB − Corr_NUTS| on active global dimensions."""
    mask_d, mask_q = batch.get('mask_d'), batch.get('mask_q')
    B = p_mb.ffx.shape[0]

    def _globals(p, b, d, q):
        return torch.cat([p.ffx[b, :, :d], p.sigma_rfx[b, :, :q], p.sigma_eps[b].unsqueeze(-1)], dim=-1)

    per_ds = []
    for b in range(B):
        d = int(mask_d[b].sum()) if mask_d is not None else p_mb.ffx.shape[-1]
        q = int(mask_q[b].sum()) if mask_q is not None else p_mb.sigma_rfx.shape[-1]
        if d + q + 1 < 2:
            continue
        try:
            c_mb   = torch.corrcoef(_globals(p_mb,   b, d, q).T)
            c_nuts = torch.corrcoef(_globals(p_nuts, b, d, q).T)
        except Exception:
            continue
        off  = ~torch.eye(c_mb.shape[0], dtype=torch.bool)
        diff = float((c_mb - c_nuts).abs()[off].mean())
        if np.isfinite(diff):
            per_ds.append(diff)

    if not per_ds:
        return {'mean |ΔCorr|': float('nan'), 'median |ΔCorr|': float('nan'), 'p90 |ΔCorr|': float('nan')}
    return {
        'mean |ΔCorr|':   float(np.mean(per_ds)),
        'median |ΔCorr|': float(np.median(per_ds)),
        'p90 |ΔCorr|':    float(np.percentile(per_ds, 90)),
    }


# ---------------------------------------------------------------------------
# Diagnostic 3 — local uncertainty decomposition (family == 0 only)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _localUncertaintyDecomp(
    p_mb: Proposal, p_nuts: Proposal, batch: dict, batch_size: int
) -> list[dict]:
    """Decompose rfx marginal std into conditional (local) vs global contributions.

    marginal_std  = std_S(b_i) / sd_y          — total, includes global uncertainty
    cond_std_Gauss = E_S[diag(Λ_i^{-1})^½] / sd_y — given globals (local only)
    ratio = cond / marginal:  near 0 = global dominates, near 1 = local dominates
    """
    group_mask = batch['mask_n'].any(-1)
    B, q = batch['y'].shape[0], p_mb.q
    sd_y = batch['sd_y']
    mask_q  = batch.get('mask_q', torch.ones(B, q, dtype=torch.bool))
    mask_mq = group_mask.unsqueeze(-1) & mask_q.unsqueeze(1)

    def _marginal_std(p):
        return p.rfx.std(dim=2) / sd_y.view(-1, 1, 1)

    def _cond_std(p):
        chunks = []
        for s in range(0, B, batch_size):
            e  = min(s + batch_size, B)
            sd = sd_y[s:e]
            sr = p.sigma_rfx[s:e] / sd.view(-1, 1, 1)
            se = p.sigma_eps[s:e]  / sd.view(-1, 1)
            Z_m   = batch['Z'][s:e, :, :, :q] * batch['mask_n'][s:e].float().unsqueeze(-1)
            ZtZ   = torch.einsum('bmnq,bmnp->bmpq', Z_m, Z_m)
            Lambda = (
                ZtZ.unsqueeze(2) / se.clamp(min=1e-6)[:, None, :, None, None] ** 2
                + torch.diag_embed(1.0 / sr.clamp(min=1e-6) ** 2).unsqueeze(1)
            )
            L     = torch.linalg.cholesky(Lambda)
            L_inv = torch.linalg.solve_triangular(
                L, torch.eye(q, device=L.device, dtype=L.dtype).expand_as(L), upper=False
            )
            chunks.append((L_inv ** 2).sum(dim=-2).sqrt().mean(dim=2))  # (bs, m, q)
        return torch.cat(chunks, dim=0)

    rows = []
    for label, proposal in [('MB', p_mb), ('NUTS', p_nuts)]:
        std_m = _marginal_std(proposal)
        std_c = _cond_std(proposal)
        ratio = std_c / std_m.clamp(min=1e-8)
        rows.append({
            'Method':               label,
            'marginal_std (med)':   float(np.median(std_m[mask_mq].numpy())),
            'cond_std_Gauss (med)': float(np.median(std_c[mask_mq].numpy())),
            'ratio p50':            float(np.median(ratio[mask_mq].numpy())),
            'ratio p25':            float(np.percentile(ratio[mask_mq].numpy(), 25)),
            'ratio p75':            float(np.percentile(ratio[mask_mq].numpy(), 75)),
        })
    return rows


# ---------------------------------------------------------------------------
# Diagnostic 4 — marginal rank calibration
# ---------------------------------------------------------------------------


def _rankDiagnostic(p_mb: Proposal, p_nuts: Proposal, batch: dict) -> list[dict]:
    """Rank of each NUTS sample within the MB marginal; should be Uniform(0,1) if shapes match.

    Expected quantiles: p10=0.10, p25=0.25, p50=0.50, p75=0.75, p90=0.90.
    """
    group_mask = batch['mask_n'].any(-1)
    mask_d, mask_q = batch.get('mask_d'), batch.get('mask_q')
    B, q = p_mb.ffx.shape[0], p_mb.rfx.shape[-1]
    mask_q_t = mask_q if mask_q is not None else torch.ones(B, q, dtype=torch.bool)

    def _rank_fracs(mb: np.ndarray, nuts: np.ndarray) -> np.ndarray:
        # mb, nuts: (N, S); returns all rank fractions flattened
        mb_sorted = np.sort(mb, axis=1)
        S = mb_sorted.shape[1]
        return np.concatenate([np.searchsorted(mb_sorted[i], nuts[i]) / S
                                for i in range(len(mb_sorted))])

    def _quantile_row(name: str, fracs: np.ndarray) -> dict:
        return {
            'Type':           name,
            'p10 (exp 0.10)': float(np.percentile(fracs, 10)),
            'p25 (exp 0.25)': float(np.percentile(fracs, 25)),
            'p50 (exp 0.50)': float(np.percentile(fracs, 50)),
            'p75 (exp 0.75)': float(np.percentile(fracs, 75)),
            'p90 (exp 0.90)': float(np.percentile(fracs, 90)),
        }

    def _global_fracs(mb_v: torch.Tensor, nuts_v: torch.Tensor, mask) -> np.ndarray:
        # mb_v, nuts_v: (B, S, d); mask: (B, d) bool or None
        mb_np, nu_np = mb_v.numpy(), nuts_v.numpy()
        d = mb_np.shape[-1]
        rows_mb, rows_nu = [], []
        for di in range(d):
            active = mask[:, di].numpy() if mask is not None else np.ones(B, dtype=bool)
            rows_mb.append(mb_np[active, :, di])
            rows_nu.append(nu_np[active, :, di])
        return _rank_fracs(np.concatenate(rows_mb), np.concatenate(rows_nu))

    rows = []
    for name, mb_v, nuts_v, mask in [
        ('ffx',       p_mb.ffx,                    p_nuts.ffx,                    mask_d),
        ('sigma_rfx', p_mb.sigma_rfx,               p_nuts.sigma_rfx,              mask_q),
        ('sigma_eps', p_mb.sigma_eps.unsqueeze(-1),  p_nuts.sigma_eps.unsqueeze(-1), None),
    ]:
        fracs = _global_fracs(mb_v, nuts_v, mask)
        if fracs.size:
            rows.append(_quantile_row(name, fracs))

    # rfx: (B, m, S, q) — pool over active (b, group, dim) triples
    mb_rfx, nu_rfx = p_mb.rfx.numpy(), p_nuts.rfx.numpy()
    rows_mb, rows_nu = [], []
    for k in range(q):
        active_bm = (group_mask & mask_q_t[:, k].unsqueeze(1)).numpy()
        rows_mb.append(mb_rfx[active_bm, :, k])
        rows_nu.append(nu_rfx[active_bm, :, k])
    fracs = _rank_fracs(np.concatenate(rows_mb), np.concatenate(rows_nu))
    if fracs.size:
        rows.append(_quantile_row('rfx', fracs))

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _table(rows, headers) -> str:
    fmt = lambda v: v if isinstance(v, str) else f'{v:.4f}'
    return tabulate([[fmt(r[h]) for h in headers] for r in rows], headers=headers, tablefmt='simple')


def _printWidth(rows: list[dict]) -> None:
    print('\n=== 1. Posterior width: std(MB) / std(NUTS) ===')
    print(_table(rows, ['Type', 'std_MB', 'std_NUTS', 'Ratio p50', 'Ratio p25', 'Ratio p75']))
    print('  Ratio > 1: MB wider (overdispersed)  |  < 1: MB narrower (underdispersed)\n')


def _printCorr(d: dict) -> None:
    print('=== 2. Global correlation structure: |Corr_MB − Corr_NUTS| ===')
    for k, v in d.items():
        print(f'  {k}: {v:.4f}')
    print('  Near 0: MB captures global correlations  |  Large: MB misses correlation structure\n')


def _printLocalUncertainty(rows: list[dict]) -> None:
    print('=== 3. Local uncertainty decomposition (rfx, standardised space) ===')
    print(_table(rows, ['Method', 'marginal_std (med)', 'cond_std_Gauss (med)', 'ratio p50', 'ratio p25', 'ratio p75']))
    print('  ratio = cond_std / marginal_std')
    print('  ratio → 0: global uncertainty dominates rfx spread')
    print('  ratio → 1: local (conditional) uncertainty dominates rfx spread\n')


def _printRank(rows: list[dict]) -> None:
    headers = ['Type', 'p10 (exp 0.10)', 'p25 (exp 0.25)', 'p50 (exp 0.50)', 'p75 (exp 0.75)', 'p90 (exp 0.90)']
    fmt = lambda v: v if isinstance(v, str) else f'{v:.3f}'
    print('=== 4. Marginal rank calibration: rank of NUTS in MB distribution ===')
    print(tabulate([[fmt(r[h]) for h in headers] for r in rows], headers=headers, tablefmt='simple'))
    print('  Values near expected → shapes match')
    print('  p50 ≠ 0.50 → bias  |  compressed quantiles → MB overdispersed  |  spread → MB underdispersed\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)

    device = setDevice(cfg.device)
    exp_dir = Path(__file__).resolve().parent

    dl_test = _loadTestData(cfg, exp_dir)
    model   = _loadModel(cfg, exp_dir, device)

    full_batch    = dl_test.fullBatch()
    batch_rescaled = rescaleData(full_batch) if cfg.rescale else full_batch
    proposal_mb   = _sampleMB(model, dl_test, cfg.n_samples, device, cfg.rescale)
    proposal_nuts = _batchToProposal(full_batch, 'nuts', cfg.rescale)

    conv_mask = nutsConvergeMask(full_batch)
    if conv_mask is not None:
        logger.info('Converged NUTS: %d / %d datasets', int(conv_mask.sum()), len(conv_mask))
        batch_rescaled = subsetBatch(batch_rescaled, conv_mask)
        proposal_mb    = subsetProposal(proposal_mb,   conv_mask)
        proposal_nuts  = subsetProposal(proposal_nuts, conv_mask)
    else:
        logger.warning('No NUTS convergence diagnostics found; using all datasets')

    _printWidth(_widthDiagnostic(proposal_mb, proposal_nuts, batch_rescaled))
    _printCorr(_corrDiagnostic(proposal_mb, proposal_nuts, batch_rescaled))

    if cfg.likelihood_family == 0:
        _printLocalUncertainty(
            _localUncertaintyDecomp(proposal_mb, proposal_nuts, batch_rescaled, cfg.batch_size)
        )
    else:
        logger.info('Local uncertainty diagnostic skipped (family != 0)')

    _printRank(_rankDiagnostic(proposal_mb, proposal_nuts, batch_rescaled))


if __name__ == '__main__':
    main()
