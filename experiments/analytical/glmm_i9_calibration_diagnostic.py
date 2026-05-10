"""Replay output-only sigma_rfx calibration schedules for I9.

This diagnostic does not call the public estimator. It traces the pre-I9 Gaussian
internals, applies candidate output-only sigma_rfx schedules, and compares sRFX
NRMSE across the required analytical benchmark suite.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from experiments.analytical.glmm_srfx_diagnostic import SIZES, _nrmse, _paths, _trace_normal
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice


_Columns = dict[str, np.ndarray]


@dataclass(frozen=True)
class _Candidate:
    name: str
    factor_fn: Callable[[_Columns], np.ndarray]


def _as_arrays(values: dict[str, list[float | str | bool]]) -> _Columns:
    arrays: _Columns = {}
    for key, vals in values.items():
        if key in {'dataset', 'path'}:
            arrays[key] = np.array(vals, dtype=object)
        elif key in {'floor', 'cap'}:
            arrays[key] = np.array(vals, dtype=bool)
        else:
            arrays[key] = np.array(vals, dtype=float)
    return arrays


def _collect_records(batch_size: int) -> _Columns:
    values: dict[str, list[float | str | bool]] = defaultdict(list)

    combos: list[tuple[str, str, int]] = []
    for size in SIZES:
        combos.append((f'{size}-n-mixed', 'train', 2))
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])

    with torch.no_grad():
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            dataset = f'{data_id}/{partition}'
            for path in _paths(data_id, partition, n_epochs):
                for batch in Dataloader(path, batch_size=batch_size, shuffle=False):
                    batch = toDevice(batch, torch.device('cpu'))
                    trace = _trace_normal(batch, max_q)
                    mask_q = batch['mask_q'][..., :max_q].bool()

                    for b in range(batch['X'].shape[0]):
                        active_q = mask_q[b]
                        if not bool(active_q.any()):
                            continue
                        active_idx = active_q.nonzero(as_tuple=False).flatten()
                        q_count = float(active_q.sum())
                        d_count = float(batch['mask_d'][b].sum())
                        G_mom = float(trace.mom_mask[b].sum())
                        path_label = 'fallback'
                        if bool(trace.enough_full_mom[b]):
                            path_label = 'full_mom'
                        elif bool(trace.enough_diag_mom[b]):
                            path_label = 'diag_mom'

                        for q_idx in active_idx.tolist():
                            component_path = path_label
                            if bool(trace.use_component_diag[b, q_idx]):
                                component_path = 'component_diag'
                            base = float(trace.sigma_final[b, q_idx])
                            floor = float(trace.psi_diag_floor[b, q_idx])
                            ratio = base * base / max(floor, 1e-12)

                            values['dataset'].append(dataset)
                            values['truth'].append(float(batch['sigma_rfx'][b, q_idx]))
                            values['base'].append(base)
                            values['q'].append(q_count)
                            values['d'].append(d_count)
                            values['G_mom'].append(G_mom)
                            values['floor_ratio'].append(ratio)
                            values['path'].append(component_path)
                            values['floor'].append(bool(trace.floor_hit_post_em[b, q_idx]))
                            values['cap'].append(bool(trace.cap_hit_post_em[b]))

    return _as_arrays(values)


def _current_i9_factor(data: _Columns) -> np.ndarray:
    out = np.ones_like(data['base'])
    floor_q_gt2 = data['floor'] & (data['q'] > 2.0)
    floor_q2_low_ratio = data['floor'] & (data['q'] == 2.0) & (data['floor_ratio'] <= 0.68)
    out = np.where(floor_q2_low_ratio, 0.85, out)
    out = np.where(floor_q_gt2, 0.55, out)
    return out


def _candidate_factors() -> list[_Candidate]:
    def no_calibration(data: _Columns) -> np.ndarray:
        return np.ones_like(data['base'])

    def q_gt2(factor: float) -> Callable[[_Columns], np.ndarray]:
        return lambda data: np.where(data['floor'] & (data['q'] > 2.0), factor, 1.0)

    def q2_plus(q2_factor: float, q_gt2_factor: float) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = np.ones_like(data['base'])
            out = np.where(data['floor'] & (data['q'] == 2.0), q2_factor, out)
            out = np.where(data['floor'] & (data['q'] > 2.0), q_gt2_factor, out)
            return out

        return factors

    def current_plus_q2_floor_ratio(
        q2_factor: float,
        ratio_max: float,
    ) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = _current_i9_factor(data)
            gate = data['floor'] & (data['q'] == 2.0) & (data['floor_ratio'] <= ratio_max)
            return np.where(gate, q2_factor, out)

        return factors

    def current_plus_q2_path(
        q2_factor: float,
        path: str,
    ) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = _current_i9_factor(data)
            gate = data['floor'] & (data['q'] == 2.0) & (data['path'] == path)
            return np.where(gate, q2_factor, out)

        return factors

    def current_plus_q2_gmom(
        q2_factor: float,
        max_gmom: float,
    ) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = _current_i9_factor(data)
            gate = data['floor'] & (data['q'] == 2.0) & (data['G_mom'] <= max_gmom)
            return np.where(gate, q2_factor, out)

        return factors

    def current_plus_q1_floor(q1_factor: float) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = _current_i9_factor(data)
            gate = data['floor'] & (data['q'] == 1.0)
            return np.where(gate, q1_factor, out)

        return factors

    def current_plus_q1_floor_ratio(
        q1_factor: float,
        ratio_max: float,
    ) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = _current_i9_factor(data)
            gate = data['floor'] & (data['q'] == 1.0) & (data['floor_ratio'] <= ratio_max)
            return np.where(gate, q1_factor, out)

        return factors

    def path_q_gt2(
        full_factor: float,
        weak_factor: float,
    ) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            floor_q = data['floor'] & (data['q'] > 2.0)
            full = data['path'] == 'full_mom'
            out = np.ones_like(data['base'])
            out = np.where(floor_q & full, full_factor, out)
            out = np.where(floor_q & ~full, weak_factor, out)
            return out

        return factors

    def current_plus_cap(cap_factor: float) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            out = _current_i9_factor(data)
            out = np.where(data['floor'] & data['cap'], np.minimum(out, cap_factor), out)
            return out

        return factors

    def weak_only(factor: float, q_min: float) -> Callable[[_Columns], np.ndarray]:
        def factors(data: _Columns) -> np.ndarray:
            weak_path = data['path'] != 'full_mom'
            gate = data['floor'] & weak_path & (data['q'] > q_min)
            return np.where(gate, factor, 1.0)

        return factors

    return [
        _Candidate('pre_i9_no_calibration', no_calibration),
        _Candidate('i9a_qgt2_f080', q_gt2(0.80)),
        _Candidate('i9b_qgt2_f070', q_gt2(0.70)),
        _Candidate('i9c_qgt2_f055', q_gt2(0.55)),
        _Candidate('i9d_current_q2_ratio_qgt2', _current_i9_factor),
        _Candidate('qgt2_f050', q_gt2(0.50)),
        _Candidate('qgt2_f055', q_gt2(0.55)),
        _Candidate('qgt2_f060', q_gt2(0.60)),
        _Candidate('qgt2_f0625', q_gt2(0.625)),
        _Candidate('qgt2_f065', q_gt2(0.65)),
        _Candidate('qgt2_f0675', q_gt2(0.675)),
        _Candidate('qgt2_f070', q_gt2(0.70)),
        _Candidate('qgt2_f075', q_gt2(0.75)),
        _Candidate('qgt2_f085', q_gt2(0.85)),
        _Candidate('q2_f095_qgt2_f070', q2_plus(0.95, 0.70)),
        _Candidate('q2_f090_qgt2_f070', q2_plus(0.90, 0.70)),
        _Candidate('q2_f085_qgt2_f070', q2_plus(0.85, 0.70)),
        _Candidate('q2_f090_ratio_le055', current_plus_q2_floor_ratio(0.90, 0.55)),
        _Candidate('q2_f090_ratio_le068', current_plus_q2_floor_ratio(0.90, 0.68)),
        _Candidate('q2_f090_ratio_le089', current_plus_q2_floor_ratio(0.90, 0.89)),
        _Candidate('q2_f085_ratio_le068', current_plus_q2_floor_ratio(0.85, 0.68)),
        _Candidate('q2_f090_full_mom', current_plus_q2_path(0.90, 'full_mom')),
        _Candidate('q2_f090_diag_mom', current_plus_q2_path(0.90, 'diag_mom')),
        _Candidate('q2_f090_gmom_le018', current_plus_q2_gmom(0.90, 18.0)),
        _Candidate('q2_f090_gmom_le028', current_plus_q2_gmom(0.90, 28.0)),
        _Candidate('q1_f090_floor', current_plus_q1_floor(0.90)),
        _Candidate('q1_f080_floor', current_plus_q1_floor(0.80)),
        _Candidate('q1_f070_floor', current_plus_q1_floor(0.70)),
        _Candidate('q1_f060_floor', current_plus_q1_floor(0.60)),
        _Candidate('q1_f080_ratio_le055', current_plus_q1_floor_ratio(0.80, 0.55)),
        _Candidate('q1_f070_ratio_le055', current_plus_q1_floor_ratio(0.70, 0.55)),
        _Candidate('q1_f080_ratio_le068', current_plus_q1_floor_ratio(0.80, 0.68)),
        _Candidate('q1_f070_ratio_le068', current_plus_q1_floor_ratio(0.70, 0.68)),
        _Candidate('q1_f080_ratio_le089', current_plus_q1_floor_ratio(0.80, 0.89)),
        _Candidate('q1_f070_ratio_le089', current_plus_q1_floor_ratio(0.70, 0.89)),
        _Candidate('qgt2_full080_weak075', path_q_gt2(0.80, 0.75)),
        _Candidate('qgt2_full085_weak075', path_q_gt2(0.85, 0.75)),
        _Candidate('qgt2_full080_weak070', path_q_gt2(0.80, 0.70)),
        _Candidate('i9_cap_floor_f075', current_plus_cap(0.75)),
        _Candidate('weak_qgt1_f090', weak_only(0.90, 1.0)),
        _Candidate('weak_qgt1_f085', weak_only(0.85, 1.0)),
    ]


def _scores_by_dataset(data: _Columns, sigma: np.ndarray) -> dict[str, float]:
    scores = {}
    for dataset in sorted(set(data['dataset'].tolist())):
        mask = data['dataset'] == dataset
        truth = data['truth'][mask]
        scores[dataset] = _nrmse(sigma[mask] - truth, truth)
    return scores


def _summarize_candidates(data: _Columns) -> tuple[list[list[str]], dict[str, dict[str, float]]]:
    candidates = _candidate_factors()
    baseline_sigma = data['base'] * _current_i9_factor(data)
    baseline_scores = _scores_by_dataset(data, baseline_sigma)
    all_scores: dict[str, dict[str, float]] = {}
    rows = []

    for candidate in candidates:
        sigma = data['base'] * candidate.factor_fn(data)
        scores = _scores_by_dataset(data, sigma)
        all_scores[candidate.name] = scores
        deltas = {key: scores[key] - baseline_scores[key] for key in scores}
        best_dataset = min(deltas, key=deltas.get)
        worst_dataset = max(deltas, key=deltas.get)
        rows.append(
            [
                candidate.name,
                f'{np.mean(list(scores.values())):.4f}',
                f'{max(scores.values()):.4f}',
                worst_dataset,
                f'{deltas[worst_dataset]:+.4f}',
                best_dataset,
                f'{deltas[best_dataset]:+.4f}',
                str(sum(delta < -1e-5 for delta in deltas.values())),
                str(sum(delta > 1e-5 for delta in deltas.values())),
            ]
        )

    rows.sort(key=lambda row: (float(row[2]), float(row[1]), int(row[8])))
    return rows, all_scores


def _label_quantile_bins(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    edges = np.unique(np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if edges.size <= 1:
        return []
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f'{lo:.2f}-{hi:.2f}'
        bins.append((label, (values >= lo) & (values <= hi)))
    return bins


def _bin_rows(data: _Columns) -> list[list[str]]:
    sigma = data['base'] * _current_i9_factor(data)
    rows = []

    def add(label: str, mask: np.ndarray) -> None:
        if not bool(mask.any()):
            return
        truth = data['truth'][mask]
        err = sigma[mask] - truth
        rel = err / np.maximum(np.abs(truth), 1e-8)
        rows.append(
            [label, str(int(mask.sum())), f'{_nrmse(err, truth):.4f}', f'{rel.mean():+.3f}']
        )

    for q_label, mask in [
        ('q=1', data['q'] == 1.0),
        ('q=2', data['q'] == 2.0),
        ('q>=3', data['q'] > 2.0),
    ]:
        add(q_label, mask)
        add(f'{q_label},floor=True', mask & data['floor'])

    for path in ['full_mom', 'diag_mom', 'component_diag', 'fallback']:
        add(f'path={path}', data['path'] == path)
        add(f'path={path},floor=True', (data['path'] == path) & data['floor'])

    for label, mask in _label_quantile_bins(data['G_mom']):
        add(f'G_mom={label}', mask)

    floor_mask = data['floor']
    if bool(floor_mask.any()):
        for label, mask in _label_quantile_bins(data['floor_ratio'][floor_mask]):
            full_mask = np.zeros_like(floor_mask, dtype=bool)
            full_mask[np.nonzero(floor_mask)[0][mask]] = True
            add(f'floor_ratio={label}', full_mask)

    add('cap=False', ~data['cap'])
    add('cap=True', data['cap'])
    add('cap=True,floor=True', data['cap'] & data['floor'])
    return rows


def run_i9_calibration_diagnostic(batch_size: int = 32) -> None:
    data = _collect_records(batch_size)
    candidate_rows, scores = _summarize_candidates(data)

    print('\nCandidate schedule ranking')
    print(
        tabulate(
            candidate_rows,
            headers=[
                'candidate',
                'mean',
                'max',
                'worst_delta_set',
                'worst_delta',
                'best_delta_set',
                'best_delta',
                'wins',
                'losses',
            ],
            tablefmt='simple',
        )
    )

    print('\nPer-dataset sRFX for top schedules')
    top = [row[0] for row in candidate_rows[:5]]
    datasets = sorted(set(data['dataset'].tolist()))
    rows = []
    for dataset in datasets:
        rows.append([dataset, *[f'{scores[name][dataset]:.4f}' for name in top]])
    print(tabulate(rows, headers=['dataset', *top], tablefmt='simple'))

    print('\nCurrent I9 bins')
    print(tabulate(_bin_rows(data), headers=['bin', 'N', 'sRFX', 'rel_bias'], tablefmt='simple'))


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = setup()
    run_i9_calibration_diagnostic(args.batch_size)
