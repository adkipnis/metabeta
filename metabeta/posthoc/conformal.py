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

