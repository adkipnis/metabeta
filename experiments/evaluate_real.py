"""
experiments/evaluate_real.py — Posterior comparison on real data: MB and ADVI vs NUTS.

Evaluates a model checkpoint on the pre-generated real-data test batch at
outputs/data/{size}-{fam}-real/test.fit.npz, comparing MB and ADVI posteriors
against NUTS as reference.  Since there are no ground-truth parameters, all
metrics are relative to NUTS; only NUTS-converged datasets are included.

Metrics (mean ± std over datasets):
  r          — Pearson r of posterior means (method vs NUTS), pooled active params
  σ-ratio    — per-dataset median(std_method / std_NUTS) across active params
  rank-MAD   — mean |empirical quantile − expected| for NUTS-in-MB rank fracs (pooled)
  ΔNLL       — LOO-NLL(method) − LOO-NLL(NUTS), per dataset
  Δtime (s)  — tpd(method) − tpd(NUTS), seconds per dataset

Usage (from experiments/):
    uv run python evaluate_real.py --checkpoint PATH
    uv run python evaluate_real.py --checkpoint PATH --prefix best --n_samples 512
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.evaluation.summary import getSummary
from metabeta.models.approximator import Approximator
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Collection, collateGrouped, subsetBatch, toDevice
from metabeta.utils.evaluation import (
    Proposal,
    concatProposalsBatch,
    nutsConvergeMask,
    subsetProposal,
)
from metabeta.utils.io import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.templates import loadConfigFromCheckpoint

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
OUT_DIR = DIR / 'results'

logger = logging.getLogger(__name__)

_FAM_LETTER = {0: 'n', 1: 'b', 2: 'p'}


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Real-data posterior comparison: MB and ADVI vs NUTS',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint',       type=str, required=True)
    parser.add_argument('--prefix',           type=str, default='latest')
    parser.add_argument('--device',           type=str, default='cpu')
    parser.add_argument('--n_samples',        type=int, default=1000)
    parser.add_argument('--batch_size',       type=int, default=8)
    parser.add_argument('--seed',             type=int, default=0)
    parser.add_argument('--outdir',           type=str, default=str(OUT_DIR))
    parser.add_argument('--verbosity',        type=int, default=1)
    parser.add_argument('--rescale',          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--convergence_mode', type=str, default='liberal',
                        choices=['liberal', 'strict'])
    # fmt: on
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading


def loadModel(
    ckpt_dir: Path,
    prefix: str,
    device: torch.device,
) -> tuple[Approximator, argparse.Namespace]:
    cfg_dict = loadConfigFromCheckpoint(ckpt_dir)
    cfg = argparse.Namespace(**cfg_dict)

    model_cfg_path = METABETA / 'configs' / 'models' / f'{cfg.model_id}.yaml'
    model_cfg = modelFromYaml(
        model_cfg_path,
        d_ffx=cfg.max_d,
        d_rfx=cfg.max_q,
        likelihood_family=cfg.likelihood_family,
    )
    model = Approximator(model_cfg).to(device)
    model.eval()

    ckpt_path = ckpt_dir / f'{prefix}.pt'
    assert ckpt_path.exists(), f'checkpoint not found: {ckpt_path}'
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])
    logger.info('Loaded %s/%s.pt', ckpt_dir.name, prefix)
    return model, cfg


# ---------------------------------------------------------------------------
# Batch / proposal helpers (shared with evaluate_oracle.py)


def fitBatchMask(batch: dict[str, torch.Tensor], prefix: str) -> np.ndarray:
    failed_key = f'{prefix}_failed'
    if failed_key not in batch:
        return np.ones(batch['X'].shape[0], dtype=bool)
    return ~batch[failed_key].cpu().numpy().astype(bool)


def fit2proposal(batch: dict[str, torch.Tensor], prefix: str) -> Proposal:
    samples_g = [batch[f'{prefix}_ffx'], batch[f'{prefix}_sigma_rfx']]
    has_sigma_eps = False
    if f'{prefix}_sigma_eps' in batch:
        samples_g.append(batch[f'{prefix}_sigma_eps'].unsqueeze(-1))
        has_sigma_eps = True
    proposed = {
        'global': {'samples': torch.cat(samples_g, dim=-1)},
        'local': {'samples': batch[f'{prefix}_rfx']},
    }
    proposal = Proposal(
        proposed,
        has_sigma_eps=has_sigma_eps,
        corr_rfx=batch.get(f'{prefix}_corr_rfx'),
    )
    proposal.tpd = batch[f'{prefix}_duration'].mean().item()
    return proposal


@torch.inference_mode()
def sampleMB(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Proposal, torch.Tensor]:
    """Sample from model; returns (proposal, tpd_arr) with tpd_arr shape (B,)."""
    B = batch['X'].shape[0]
    proposals: list[Proposal] = []
    tpd_list: list[float] = []
    for start in tqdm(range(0, B, batch_size), desc='  MB', leave=False):
        end = min(start + batch_size, B)
        b_chunk = {
            k: v[start:end] if (torch.is_tensor(v) and v.shape[0] == B) else v
            for k, v in batch.items()
        }
        b_chunk = toDevice(b_chunk, device)
        t0 = time.perf_counter()
        p_chunk = model.estimate(b_chunk, n_samples=n_samples)
        tpd_list.extend([(time.perf_counter() - t0) / (end - start)] * (end - start))
        p_chunk.to('cpu')
        proposals.append(p_chunk)
    merged = concatProposalsBatch(proposals)
    tpd_arr = torch.tensor(tpd_list, dtype=torch.float64)
    merged.tpd = tpd_arr.mean().item()
    return merged, tpd_arr


# ---------------------------------------------------------------------------
# Per-dataset posterior comparison metrics


def _masks(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Returns (mask_d, mask_q, group_mask) of shapes (B,max_d), (B,max_q), (B,max_m)."""
    return (
        batch.get('mask_d'),
        batch.get('mask_q'),
        batch['mask_n'].any(-1),
    )


def _pooledMeans(
    p: Proposal,
    b: int,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    group_mask: torch.Tensor,
) -> np.ndarray:
    """Posterior means for all active params of dataset b as a flat numpy array."""
    d_mask = mask_d[b] if mask_d is not None else torch.ones(p.d, dtype=torch.bool)
    q_mask = mask_q[b] if mask_q is not None else torch.ones(p.q, dtype=torch.bool)

    # rfx[b]: (max_m, S, max_q) — mean over S (dim 1) → (max_m, max_q)
    mean_rfx = p.rfx[b].mean(1)[group_mask[b]][:, q_mask].ravel()

    parts = [
        p.ffx[b].mean(0)[d_mask].numpy(),
        p.sigma_rfx[b].mean(0)[q_mask].numpy(),
        mean_rfx.numpy(),
    ]
    if p.has_sigma_eps:
        parts.append(p.sigma_eps[b].mean().reshape(1).numpy())
    return np.concatenate(parts)


def computeCorr(
    p_method: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
) -> np.ndarray:
    """Pearson r between all active posterior means (method vs NUTS) per dataset.

    Returns (B,) float array; NaN when fewer than 2 active params.
    """
    B = p_method.ffx.shape[0]
    mask_d, mask_q, group_mask = _masks(batch)
    r_vals = np.empty(B)
    for b in range(B):
        v_m = _pooledMeans(p_method, b, mask_d, mask_q, group_mask)
        v_n = _pooledMeans(p_nuts, b, mask_d, mask_q, group_mask)
        r_vals[b] = np.corrcoef(v_m, v_n)[0, 1] if len(v_m) >= 2 else np.nan
    return r_vals


def _stdRatios(
    p_method: Proposal,
    p_nuts: Proposal,
    b: int,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    group_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-active-entry std_method / std_nuts for dataset b as a flat tensor."""
    d_mask = mask_d[b] if mask_d is not None else torch.ones(p_method.d, dtype=torch.bool)
    q_mask = mask_q[b] if mask_q is not None else torch.ones(p_method.q, dtype=torch.bool)

    def _ratio(a: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
        return a / b_.clamp(min=1e-8)

    # rfx[b]: (max_m, S, max_q) — std over S (dim 1) → (max_m, max_q), then select active
    rfx_std_m = p_method.rfx[b][group_mask[b]].std(1)[:, q_mask]
    rfx_std_n = p_nuts.rfx[b][group_mask[b]].std(1)[:, q_mask]

    parts: list[torch.Tensor] = [
        _ratio(p_method.ffx[b].std(0)[d_mask], p_nuts.ffx[b].std(0)[d_mask]),
        _ratio(p_method.sigma_rfx[b].std(0)[q_mask], p_nuts.sigma_rfx[b].std(0)[q_mask]),
        _ratio(rfx_std_m, rfx_std_n).reshape(-1),
    ]
    if p_method.has_sigma_eps:
        parts.append(
            _ratio(
                p_method.sigma_eps[b].std(dim=0, keepdim=True),
                p_nuts.sigma_eps[b].std(dim=0, keepdim=True),
            )
        )
    return torch.cat(parts)


def computeSigmaRatio(
    p_method: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
) -> np.ndarray:
    """Per-dataset median(std_method / std_nuts) across all active params. Returns (B,)."""
    B = p_method.ffx.shape[0]
    mask_d, mask_q, group_mask = _masks(batch)
    ratios = np.empty(B)
    for b in range(B):
        vals = _stdRatios(p_method, p_nuts, b, mask_d, mask_q, group_mask)
        ratios[b] = float(vals.median()) if vals.numel() > 0 else np.nan
    return ratios


def _rankFracs(mb: np.ndarray, nuts: np.ndarray) -> np.ndarray:
    """Rank of each NUTS sample within the corresponding MB marginal, as a fraction.

    mb, nuts: (N, S) — N active entries, S samples.  Returns all fracs flattened.
    """
    mb_sorted = np.sort(mb, axis=1)
    S = mb_sorted.shape[1]
    return np.concatenate(
        [np.searchsorted(mb_sorted[i], nuts[i]) / S for i in range(len(mb_sorted))]
    )


def computeRankMAD(
    p_method: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
) -> float:
    """Mean |empirical quantile − expected| for NUTS-in-method rank fracs (pooled).

    Fracs are pooled over all active (param_dim × dataset) entries.
    Returns 0 for perfect shape agreement, larger values for systematic disagreement.
    """
    mask_d, mask_q, group_mask = _masks(batch)
    B = p_method.ffx.shape[0]
    all_fracs: list[np.ndarray] = []

    def _add_global(mb_v: torch.Tensor, nu_v: torch.Tensor, mask: torch.Tensor | None) -> None:
        mb_np, nu_np = mb_v.numpy(), nu_v.numpy()   # (B, S, D)
        D = mb_np.shape[-1]
        for di in range(D):
            active = mask[:, di].numpy() if mask is not None else np.ones(B, dtype=bool)
            if active.any():
                all_fracs.append(_rankFracs(mb_np[active, :, di], nu_np[active, :, di]))

    _add_global(p_method.ffx, p_nuts.ffx, mask_d)
    _add_global(p_method.sigma_rfx, p_nuts.sigma_rfx, mask_q)
    if p_method.has_sigma_eps:
        _add_global(p_method.sigma_eps.unsqueeze(-1), p_nuts.sigma_eps.unsqueeze(-1), None)

    # rfx: (B, max_m, S, max_q) — pool over active (b, m) pairs per rfx dim
    mb_rfx = p_method.rfx.numpy()
    nu_rfx = p_nuts.rfx.numpy()
    q_any = int(mask_q.any(0).sum()) if mask_q is not None else p_method.rfx.shape[-1]
    gm_np = group_mask.numpy()

    for k in range(q_any):
        qk_active = mask_q[:, k].numpy() if mask_q is not None else np.ones(B, dtype=bool)
        bm_active = gm_np & qk_active[:, None]  # (B, max_m) bool
        if bm_active.any():
            mb_k = mb_rfx[bm_active][:, :, k]  # (n_active, S)
            nu_k = nu_rfx[bm_active][:, :, k]
            all_fracs.append(_rankFracs(mb_k, nu_k))

    if not all_fracs:
        return float('nan')

    fracs = np.concatenate(all_fracs)
    expected = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    empirical = np.array([np.percentile(fracs, 100 * q) for q in expected])
    return float(np.mean(np.abs(empirical - expected)))


# ---------------------------------------------------------------------------
# Row assembly


def _ms(arr: np.ndarray | None) -> tuple[float, float] | None:
    """Mean and Bessel-corrected std, NaNs ignored. Returns None when arr is None."""
    if arr is None:
        return None
    a = arr[~np.isnan(arr)]
    if len(a) == 0:
        return (float('nan'), float('nan'))
    return (float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)


def _buildRow(
    label: str,
    r: np.ndarray,
    sigma_ratio: np.ndarray,
    rank_mad: float,
    delta_nll: np.ndarray,
    delta_tpd: np.ndarray | None,
) -> dict:
    return {
        'method': label,
        'r': _ms(r),
        'sigma_ratio': _ms(sigma_ratio),
        'rank_mad': rank_mad,
        'delta_nll': _ms(delta_nll),
        'delta_tpd': _ms(delta_tpd),
    }


# ---------------------------------------------------------------------------
# Main evaluation


def evaluateReal(
    model: Approximator,
    data_path: Path,
    max_d: int,
    max_q: int,
    lf: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    rescale: bool,
    convergence_mode: str,
) -> list[dict]:
    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B_total = len(col)
    batch = collateGrouped([col[i] for i in range(B_total)])

    # Restrict to NUTS-converged datasets
    conv_mask = nutsConvergeMask(batch, mode=convergence_mode)
    if conv_mask is not None:
        n_conv = int(conv_mask.sum())
        logger.info('NUTS convergence (%s): %d / %d', convergence_mode, n_conv, B_total)
        if n_conv == 0:
            logger.warning('No converged datasets; aborting.')
            return []
        batch = subsetBatch(batch, conv_mask)
    else:
        logger.warning('No NUTS convergence diagnostics found; using all %d datasets', B_total)

    B = batch['X'].shape[0]

    # ADVI subset (some fits may have failed)
    advi_mask = fitBatchMask(batch, 'advi')
    n_advi = int(advi_mask.sum())
    logger.info('ADVI available: %d / %d', n_advi, B)
    advi_batch: dict | None = subsetBatch(batch, advi_mask) if n_advi > 0 else None

    # Inference
    proposal_mb, mb_tpd_arr = sampleMB(model, batch, n_samples, batch_size, device)
    proposal_nuts = fit2proposal(batch, 'nuts')
    proposal_advi = fit2proposal(advi_batch, 'advi') if advi_batch is not None else None

    # Rescale all to original data space before metric computation
    if rescale:
        proposal_mb.rescale(batch['sd_y'])
        proposal_nuts.rescale(batch['sd_y'])
        if proposal_advi is not None:
            proposal_advi.rescale(advi_batch['sd_y'])
        batch = rescaleData(batch)
        if advi_batch is not None:
            advi_batch = rescaleData(advi_batch)

    # LOO-NLL via getSummary; NRMSE/corr will be NaN since real data has no ground truth
    summary_mb = getSummary(proposal_mb, batch, lf, compute_pred_coverage=False)
    summary_nuts = getSummary(proposal_nuts, batch, lf, compute_pred_coverage=False)
    summary_advi = (
        getSummary(proposal_advi, advi_batch, lf, compute_pred_coverage=False)
        if proposal_advi is not None
        else None
    )

    nuts_tpd_arr = batch.get('nuts_duration')            # (B,) tensor or None
    advi_mask_t = torch.from_numpy(advi_mask)           # bool tensor for indexing

    rows: list[dict] = []

    for label, p_method, batch_sub, tpd_arr, summary in [
        (
            'MB',
            proposal_mb,
            batch,
            mb_tpd_arr,
            summary_mb,
        ),
        (
            'ADVI',
            proposal_advi,
            advi_batch,
            advi_batch.get('advi_duration') if advi_batch is not None else None,
            summary_advi,
        ),
    ]:
        if p_method is None or summary is None:
            continue

        is_advi = label == 'ADVI'

        # NUTS references restricted to this method's subset
        p_nuts_ref = subsetProposal(proposal_nuts, advi_mask) if is_advi else proposal_nuts
        nuts_nll = summary_nuts.loo_nll[advi_mask_t] if is_advi else summary_nuts.loo_nll
        nuts_tpd = (
            nuts_tpd_arr[advi_mask_t] if (nuts_tpd_arr is not None and is_advi) else nuts_tpd_arr
        )

        delta_tpd: np.ndarray | None = None
        if tpd_arr is not None and nuts_tpd is not None:
            delta_tpd = (tpd_arr.float() - nuts_tpd.float()).numpy()

        rows.append(
            _buildRow(
                label=label,
                r=computeCorr(p_method, p_nuts_ref, batch_sub),
                sigma_ratio=computeSigmaRatio(p_method, p_nuts_ref, batch_sub),
                rank_mad=computeRankMAD(p_method, p_nuts_ref, batch_sub),
                delta_nll=(summary.loo_nll - nuts_nll).float().numpy(),
                delta_tpd=delta_tpd,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Table output

METRICS = ['r', 'sigma_ratio', 'rank_mad', 'delta_nll', 'delta_tpd']

HEADERS_MD = ['method', 'r ↑', 'σ-ratio → 1', 'rank-MAD ↓', 'ΔNLL ↓', 'Δtime (s) ↓']

HEADERS_TEX = [
    r'$\mathrm{model}$',
    r'$r$',
    r'$\sigma\text{-ratio}$',
    r'$\mathrm{rank\text{-}MAD}$',
    r'$\Delta\mathrm{NLL}$',
    r'$\Delta\mathrm{time}\ (\mathrm{s})$',
]


def _fmtMd(val: tuple[float, float] | float | None) -> str:
    if val is None:
        return 'NA'
    if isinstance(val, tuple):
        m, s = val
        return 'NA' if m != m else f'{m:.3f} ± {s:.3f}'
    return f'{val:.4f}' if val == val else 'NA'


def _fmtTex(val: tuple[float, float] | float | None) -> str:
    if val is None:
        return r'\textrm{NA}'
    if isinstance(val, tuple):
        m, s = val
        return r'\textrm{NA}' if m != m else f'${m:.3f} \\pm {s:.3f}$'
    return f'${val:.4f}$' if val == val else r'\textrm{NA}'


def saveTables(rows: list[dict], outdir: Path, run_name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Markdown ---
    md_rows = [[r['method']] + [_fmtMd(r[m]) for m in METRICS] for r in rows]
    md_table = tabulate(md_rows, headers=HEADERS_MD, tablefmt='pipe', stralign='right')
    md_path = outdir / f'real_{run_name}.md'
    md_path.write_text(f'# Real-data evaluation: {run_name}\n\n{md_table}\n')
    logger.info('Saved Markdown → %s', md_path)

    # --- LaTeX ---
    header_row = ' & '.join(HEADERS_TEX) + r' \\'
    lines: list[str] = [
        r'\begin{tabular}{c|ccccc}',
        r'    \toprule',
        f'    {header_row}',
        r'    \midrule',
    ]
    for row in rows:
        cells = ' & '.join([rf'\texttt{{{row["method"]}}}'] + [_fmtTex(row[m]) for m in METRICS])
        lines.append(f'    {cells} \\\\')
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    tex_path = outdir / f'real_{run_name}.tex'
    tex_path.write_text('\n'.join(lines))
    logger.info('Saved LaTeX → %s', tex_path)


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)

    ckpt_dir = Path(cfg.checkpoint)
    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)

    size = model_cfg.data_id.split('-')[0]
    lf = model_cfg.likelihood_family
    fam = _FAM_LETTER[lf]

    data_id = f'{size}-{fam}-real'
    data_path = METABETA / 'outputs' / 'data' / data_id / 'test.fit.npz'
    if not data_path.exists():
        logger.error('Test file not found: %s', data_path)
        return

    logger.info('Checkpoint: %s  (prefix=%s)', ckpt_dir.name, cfg.prefix)
    logger.info('Data      : %s', data_path)
    logger.info('max_d=%d  max_q=%d  family=%d', model_cfg.max_d, model_cfg.max_q, lf)

    rows = evaluateReal(
        model=model,
        data_path=data_path,
        max_d=model_cfg.max_d,
        max_q=model_cfg.max_q,
        lf=lf,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        device=device,
        rescale=cfg.rescale,
        convergence_mode=cfg.convergence_mode,
    )

    if not rows:
        logger.error('No rows produced — check data and convergence settings.')
        return

    md_rows = [[r['method']] + [_fmtMd(r[m]) for m in METRICS] for r in rows]
    print('\n' + tabulate(md_rows, headers=HEADERS_MD, tablefmt='simple'))

    saveTables(rows, Path(cfg.outdir), ckpt_dir.name)


if __name__ == '__main__':
    main()
