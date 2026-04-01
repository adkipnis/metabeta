import numpy as np
import torch
from pathlib import Path

from metabeta.evaluation.intervals import ALPHAS, getCredibleIntervals
from metabeta.utils.evaluation import Proposal, getMasks

CIDict = dict[str, torch.Tensor]
Corrections = dict[float, CIDict]


class Calibrator:
    def __init__(self, alphas: list[float] = ALPHAS):
        self.alphas = list(alphas)
        self.corrections: Corrections = {}

    def __repr__(self) -> str:
        return str(self.corrections)

    def apply(self, ci_dicts: dict[float, CIDict]) -> dict[float, CIDict]:
        out = {}
        for alpha, ci in ci_dicts.items():
            deltas = self.corrections.get(alpha, {})
            out[alpha] = {}
            for key, quantiles in ci.items():
                delta = deltas.get(key)
                if delta is None:
                    out[alpha][key] = quantiles
                    continue
                q_c = quantiles.clone()
                if quantiles.dim() == 2:  # sigma_eps: (b, 2)
                    q_c[..., 0] -= delta
                    q_c[..., 1] += delta
                else:
                    q_c[..., 0, :] -= delta
                    q_c[..., 1, :] += delta
                out[alpha][key] = q_c
        return out

    def calibrate(self, proposal: Proposal, data: dict[str, torch.Tensor]) -> None:
        ci_dicts = getCredibleIntervals(proposal, self.alphas)
        masks = getMasks(data)
        targets = {
            'ffx': data['ffx'],
            'sigma_rfx': data['sigma_rfx'],
            'rfx': data['rfx'],
        }
        if 'sigma_eps' in data:
            targets['sigma_eps'] = data['sigma_eps']
        for alpha, ci in ci_dicts.items():
            self.corrections[alpha] = {}
            for key, quantiles in ci.items():
                self.corrections[alpha][key] = self.calibrateSingle(
                    quantiles, targets[key], alpha, masks[key]
                )

    def calibrateSingle(
        self,
        quantiles: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # sigma_eps arrives as (b, 2); unsqueeze to uniform (*, 2, d) path
        scalar = quantiles.dim() == 2
        if scalar:
            quantiles = quantiles.unsqueeze(-1)  # (b, 2, 1)
            targets = targets.unsqueeze(-1)  # (b, 1)

        b = targets.shape[0]
        alpha_t = torch.tensor(alpha)

        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        elif scalar:
            mask = mask.unsqueeze(-1)

        # transpose to (..., d, 2) for score indexing
        q = quantiles.movedim(-2, -1)
        scores = torch.zeros_like(q)

        # non-conformity scores: positive means outside the interval
        scores[..., 0] = q[..., 0] - targets  # lower: q_low - target
        scores[..., 1] = targets - q[..., 1]  # upper: target - q_high

        # get the score for the closer boundary (equivalent to max(q_low - y, y - q_high))
        midx = scores.abs().min(-1)[1].unsqueeze(-1)
        scores = torch.gather(scores, -1, midx).squeeze(-1)  # (b, ..., d)
        scores[~mask] = float('inf')  # exclude inactive dimensions from quantile
        B = mask.sum(0)

        # sort and get the (1 - alpha) empirical quantile of scores
        scores, _ = scores.sort(0, descending=False)
        factor = mask.float().mean(0) * b * 1.01
        B_safe = B.clamp(min=1) - 1
        idx = (factor * (1 - alpha_t)).ceil().clamp(min=torch.tensor(0), max=B_safe).to(torch.int64)
        correction = torch.gather(scores, dim=0, index=idx.unsqueeze(0)).squeeze(0)
        correction[B == 0] = 0.0  # no correction for fully-masked dimensions

        # average over groups for rfx (quantiles: (b, m, 2, q))
        if quantiles.dim() == 4:
            correction = correction.mean(0)

        return correction.squeeze() if scalar else correction

    def insert(self, corrections: Corrections) -> None:
        self.corrections = corrections
        self.alphas = list(corrections.keys())

    def save(self, model_id: str, suffix: str = '') -> None:
        ckpt_dir = Path(__file__).resolve().parent.parent / 'outputs' / 'checkpoints' / model_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fn = Path(ckpt_dir, f'calibrator{suffix}.npz')
        # flatten nested dict to "{alpha}/{key}" string keys
        out = {
            f'{a}/{k}': t.cpu().numpy()
            for a, per_alpha in self.corrections.items()
            for k, t in per_alpha.items()
        }
        np.savez_compressed(fn, **out, allow_pickle=True)
        print(f'Saved calibration values to {fn}.')

    def load(self, model_id: str, suffix: str = '') -> None:
        fn = (
            Path(__file__).resolve().parent.parent
            / 'outputs'
            / 'checkpoints'
            / model_id
            / f'calibrator{suffix}.npz'
        )
        raw = np.load(fn)
        corrections: Corrections = {}
        for key in raw.files:
            alpha_str, param = key.split('/', 1)
            alpha = float(alpha_str)
            if alpha not in corrections:
                corrections[alpha] = {}
            corrections[alpha][param] = torch.from_numpy(raw[key])
        self.insert(corrections)
        print(f'Loaded calibration values from {fn}.')
