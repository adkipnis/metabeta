"""
Oracle evaluation: given a model checkpoint, evaluate on tiny/small/medium/large test sets.

Filters datasets that exceed the model's d/q capacity, loads NUTS/ADVI fits from the
test.fit.npz batch, and produces a LaTeX + Markdown table with mean ± std over
parameter dimensions (for NRMSE/ECE/EACE/R) and over datasets (for LOO-NLL).

Usage (from experiments/):
    uv run python evaluate_oracle.py --checkpoint PATH
    uv run python evaluate_oracle.py --checkpoint PATH --n_samples 100 --batch_size 4
    uv run python evaluate_oracle.py --checkpoint PATH --data_ids small-n-sampled large-n-sampled
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Collection, collateGrouped, subsetBatch, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch
from metabeta.utils.io import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.sampling import setSeed
from metabeta.utils.templates import loadConfigFromCheckpoint
from metabeta.evaluation.summary import getSummary

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
OUT_DIR = DIR / 'results'

DEFAULT_DATA_IDS = [
    'tiny-n-sampled',
    'small-n-sampled',
    'medium-n-sampled',
    'large-n-sampled',
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Oracle cross-size evaluation', argument_default=argparse.SUPPRESS
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prefix',     type=str, default='latest')
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--n_samples',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--data_ids',   type=str, nargs='+', default=DEFAULT_DATA_IDS)
    parser.add_argument('--outdir',     type=str, default=str(OUT_DIR))
    parser.add_argument('--verbosity',  type=int, default=1)
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

    return model, cfg


# ---------------------------------------------------------------------------
# Batch helpers


def capacityMask(batch: dict[str, torch.Tensor], max_d: int, max_q: int) -> np.ndarray:
    d_active = batch['mask_d'].sum(-1).numpy()
    q_active = batch['mask_q'].sum(-1).numpy()
    return (d_active <= max_d) & (q_active <= max_q)


def trimBatch(
    batch: dict[str, torch.Tensor], max_d: int, max_q: int
) -> dict[str, torch.Tensor]:
    """Slice all relevant tensors to model's max_d/max_q and recompute derived masks.

    Safe because permute=False ensures features are in natural (ascending) order,
    so slicing to max_d preserves exactly the active dimensions.
    """
    out = dict(batch)

    for key in ('X', 'ffx', 'nu_ffx', 'tau_ffx', 'mask_d'):
        if key in out:
            out[key] = out[key][..., :max_d]

    for key in ('Z', 'sigma_rfx', 'tau_rfx', 'mask_q'):
        if key in out:
            out[key] = out[key][..., :max_q]

    if 'rfx' in out:
        out['rfx'] = out['rfx'][..., :max_q]

    if 'corr_rfx' in out:
        out['corr_rfx'] = out['corr_rfx'][..., :max_q, :max_q]

    for method in ('nuts', 'advi'):
        if f'{method}_ffx' in out:
            out[f'{method}_ffx'] = out[f'{method}_ffx'][..., :max_d]
        if f'{method}_sigma_rfx' in out:
            out[f'{method}_sigma_rfx'] = out[f'{method}_sigma_rfx'][..., :max_q]
        if f'{method}_rfx' in out:
            out[f'{method}_rfx'] = out[f'{method}_rfx'][..., :max_q]
        if f'{method}_corr_rfx' in out:
            out[f'{method}_corr_rfx'] = out[f'{method}_corr_rfx'][..., :max_q, :max_q]

    # recompute masks that depend on mask_q
    B = out['mask_q'].shape[0]
    out['mask_mq'] = out['mask_m'].unsqueeze(-1) & out['mask_q'].unsqueeze(-2)
    q = max_q
    out['mask_corr'] = (
        torch.stack(
            [
                out['mask_q'][..., i] & out['mask_q'][..., j]
                for i in range(1, q)
                for j in range(i)
            ],
            dim=-1,
        )
        if q >= 2
        else out['mask_q'].new_zeros(B, 0)
    )

    return out


def loadRegimeBatch(
    data_path: Path,
    max_d: int,
    max_q: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Load test batch, filtering and padding/trimming to model capacity.

    When the test set fits within the model (d_file ≤ max_d, q_file ≤ max_q),
    loads with max_d/max_q so the model receives correctly-padded inputs.
    Otherwise loads natively, filters datasets by capacity, and trims to max_d/max_q.

    Returns (batch, n_total, n_kept).
    """
    col = Collection(data_path, permute=False)
    d_file, q_file = col.d, col.q
    n_total = len(col)

    if d_file <= max_d and q_file <= max_q:
        col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
        batch = collateGrouped([col[i] for i in range(n_total)])
        return batch, n_total, n_total

    # Some datasets exceed capacity: load natively, filter, trim
    batch = collateGrouped([col[i] for i in range(n_total)])
    cap_mask = capacityMask(batch, max_d, max_q)
    n_kept = int(cap_mask.sum())
    batch = subsetBatch(batch, cap_mask)
    batch = trimBatch(batch, max_d, max_q)
    return batch, n_total, n_kept


# ---------------------------------------------------------------------------
# Inference helpers


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
    corr_rfx = batch.get(f'{prefix}_corr_rfx', None)
    proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps, corr_rfx=corr_rfx)
    proposal.tpd = batch[f'{prefix}_duration'].mean().item()
    return proposal


@torch.inference_mode()
def sampleMB(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> Proposal:
    B = batch['X'].shape[0]
    proposals: list[Proposal] = []
    t0 = time.perf_counter()
    for start in tqdm(range(0, B, batch_size), desc='  MB', leave=False):
        end = min(start + batch_size, B)
        b_chunk = {
            k: v[start:end] if (torch.is_tensor(v) and v.shape[0] == B) else v
            for k, v in batch.items()
        }
        b_chunk = toDevice(b_chunk, device)
        p_chunk = model.estimate(b_chunk, n_samples=n_samples)
        p_chunk.to('cpu')
        proposals.append(p_chunk)
    elapsed = time.perf_counter() - t0
    merged = concatProposalsBatch(proposals)
    merged.tpd = elapsed / max(B, 1)
    return merged


# ---------------------------------------------------------------------------
# Metric helpers


def flattenActiveParams(
    metric_dict: dict[str, torch.Tensor],
    active_d: torch.Tensor,
    active_q: torch.Tensor,
    has_eps: bool,
) -> torch.Tensor:
    """Flatten per-parameter-dimension metrics to a 1-D tensor over active dims only.

    Handles ffx (d,), sigma_rfx (q,), rfx (q,), sigma_eps (scalar).
    Excludes corr_rfx.
    """
    parts: list[torch.Tensor] = []
    if 'ffx' in metric_dict:
        parts.append(metric_dict['ffx'][active_d].float())
    if 'sigma_rfx' in metric_dict:
        parts.append(metric_dict['sigma_rfx'][active_q].float())
    if 'rfx' in metric_dict:
        parts.append(metric_dict['rfx'][active_q].float())
    if has_eps and 'sigma_eps' in metric_dict:
        val = metric_dict['sigma_eps'].float()
        parts.append(val.reshape(1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _ms(t: torch.Tensor) -> tuple[float, float]:
    """Mean and Bessel-corrected std, ignoring NaNs."""
    t = t[~torch.isnan(t)]
    if len(t) == 0:
        return float('nan'), float('nan')
    mean = t.mean().item()
    std = t.std(correction=1).item() if len(t) > 1 else 0.0
    return mean, std


def buildRow(
    label: str,
    regime: str,
    corr_vals: torch.Tensor,
    nrmse_vals: torch.Tensor,
    ece_vals: torch.Tensor,
    eace_vals: torch.Tensor,
    loo_nll: torch.Tensor | None,
    tpd: float | None,
) -> dict:
    row: dict = {'regime': regime, 'method': label}
    row['r'] = _ms(corr_vals)
    row['NRMSE'] = _ms(nrmse_vals)
    row['ECE'] = _ms(ece_vals)
    row['EACE'] = _ms(eace_vals)
    row['LOO-NLL'] = _ms(loo_nll) if loo_nll is not None else (float('nan'), float('nan'))
    row['time'] = tpd
    return row


# ---------------------------------------------------------------------------
# Regime evaluation


def evaluateRegime(
    model: Approximator,
    data_path: Path,
    max_d: int,
    max_q: int,
    likelihood_family: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    regime: str,
) -> list[dict]:
    logger.info('\n--- Regime: %s ---', regime)

    cap_batch, n_total, n_kept = loadRegimeBatch(data_path, max_d, max_q)
    logger.info('  Capacity filter: %d / %d (d≤%d, q≤%d)', n_kept, n_total, max_d, max_q)
    if n_kept == 0:
        logger.warning('  No datasets pass capacity filter — skipping.')
        return []

    advi_mask = fitBatchMask(cap_batch, 'advi')
    n_advi = int(advi_mask.sum())
    logger.info('  ADVI success: %d / %d', n_advi, n_kept)
    advi_batch = subsetBatch(cap_batch, advi_mask)

    proposal_mb = sampleMB(model, cap_batch, n_samples, batch_size, device)
    proposal_nuts = fit2proposal(cap_batch, 'nuts')
    proposal_advi = fit2proposal(advi_batch, 'advi') if n_advi > 0 else None

    rows = []
    for label, proposal, batch in [
        ('MB', proposal_mb, cap_batch),
        ('NUTS', proposal_nuts, cap_batch),
        ('ADVI', proposal_advi, advi_batch),
    ]:
        if proposal is None:
            continue

        proposal.to('cpu')
        summary = getSummary(
            proposal, batch, likelihood_family=likelihood_family, compute_pred_coverage=False
        )

        active_d = batch['mask_d'].any(0)
        active_q = batch['mask_q'].any(0)
        has_eps = 'sigma_eps' in summary.nrmse

        rows.append(buildRow(
            label,
            regime,
            corr_vals=flattenActiveParams(summary.corr, active_d, active_q, has_eps),
            nrmse_vals=flattenActiveParams(summary.nrmse, active_d, active_q, has_eps),
            ece_vals=flattenActiveParams(summary.ece, active_d, active_q, has_eps),
            eace_vals=flattenActiveParams(summary.eace, active_d, active_q, has_eps),
            loo_nll=summary.loo_nll,
            tpd=summary.tpd,
        ))

    return rows


# ---------------------------------------------------------------------------
# Table output

METRICS = ['r', 'NRMSE', 'ECE', 'EACE', 'LOO-NLL', 'time']


def _fmtMd(val: tuple[float, float] | float | None) -> str:
    if val is None:
        return 'NA'
    if isinstance(val, tuple):
        m, s = val
        if m != m:  # NaN check
            return 'NA'
        return f'{m:.3f} ± {s:.3f}'
    return f'{val:.4f}'


def _fmtTex(val: tuple[float, float] | float | None) -> str:
    if val is None:
        return 'NA'
    if isinstance(val, tuple):
        m, s = val
        if m != m:  # NaN check
            return 'NA'
        return f'${m:.3f} \\pm {s:.3f}$'
    return f'{val:.4f}'


def saveTables(
    rows_by_regime: dict[str, list[dict]],
    outdir: Path,
    run_name: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Markdown ---
    md_rows = []
    for regime, rows in rows_by_regime.items():
        for r in rows:
            md_rows.append([regime, r['method']] + [_fmtMd(r[c]) for c in METRICS])
    md_table = tabulate(
        md_rows,
        headers=['regime', 'method'] + METRICS,
        tablefmt='pipe',
        stralign='right',
    )
    md_path = outdir / f'oracle_{run_name}.md'
    md_path.write_text(f'# Oracle Evaluation: {run_name}\n\n{md_table}\n')
    logger.info('Saved Markdown → %s', md_path)

    # --- LaTeX ---
    header_cols = (
        r'$r$ & $\mathrm{NRMSE}$ & $\mathrm{ECE}$ & '
        r'$\mathrm{EACE}$ & $\mathrm{LOO\text{-}NLL}$ & $\mathrm{time}$'
    )
    lines: list[str] = [
        r'\begin{tabular}{cc|cccccc}',
        r'    \toprule',
        rf'    $\mathrm{{regime}}$ & $\mathrm{{model}}$ & {header_cols} \\',
    ]
    for regime, rows in rows_by_regime.items():
        lines.append(r'    \midrule')
        for j, row in enumerate(rows):
            regime_cell = rf'\texttt{{{regime}}}' if j == 0 else ''
            method_cell = rf'\texttt{{{row["method"]}}}'
            cells = ' & '.join(_fmtTex(row[c]) for c in METRICS)
            lines.append(rf'      {regime_cell} & {method_cell} & {cells} \\')
    lines += [r'    \bottomrule', r'\end{tabular}', '']

    tex_path = outdir / f'oracle_{run_name}.tex'
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
    model, model_cfg_ns = loadModel(ckpt_dir, cfg.prefix, device)
    max_d: int = model_cfg_ns.max_d
    max_q: int = model_cfg_ns.max_q
    lf: int = model_cfg_ns.likelihood_family
    run_name = ckpt_dir.name

    logger.info('Model: %s  max_d=%d  max_q=%d  likelihood=%d', run_name, max_d, max_q, lf)

    rows_by_regime: dict[str, list[dict]] = {}
    for data_id in cfg.data_ids:
        data_path = METABETA / 'outputs' / 'data' / data_id / 'test.fit.npz'
        if not data_path.exists():
            logger.warning('Skipping %s: test.fit.npz not found', data_id)
            continue
        regime = data_id.split('-')[0]
        rows = evaluateRegime(
            model,
            data_path,
            max_d,
            max_q,
            lf,
            n_samples=cfg.n_samples,
            batch_size=cfg.batch_size,
            device=device,
            regime=regime,
        )
        if rows:
            rows_by_regime[regime] = rows

    if not rows_by_regime:
        logger.error('No regimes evaluated — check data_ids and checkpoint.')
        return

    saveTables(rows_by_regime, Path(cfg.outdir), run_name)


if __name__ == '__main__':
    main()
