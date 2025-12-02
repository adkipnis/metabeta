from collections.abc import Iterable
import torch
from metabeta.utils import palette
from pathlib import Path


CI = [50, 68, 80, 90, 95]

class Calibrator:
    def __init__(self, d: int, intervals: Iterable[int] = range(1, 100)):
        self.d = d  # dimensionality of target
        self.intervals = intervals
        self.corrections = {str(i): torch.zeros(d) for i in self.intervals}

    def __repr__(self) -> str:
        return str(self.corrections)

    def update(self, Q: torch.Tensor, i: int) -> None:
        assert Q.dim() == 1 and len(Q) == self.d
        self.corrections[str(i)] = Q

    def get(self, i: int) -> torch.Tensor:
        return self.corrections[str(i)]

    def calibrate(
        self,
        model,
        proposed: dict[str, torch.Tensor],
        targets: torch.Tensor,
        local: bool = False,
    ) -> None:
        # wrapper for calibrateSingle()
        for i in self.intervals:
            alpha = (100 - i) / 100
            quantiles = model.quantiles(
                proposed, [alpha / 2, 1 - alpha / 2], local=local, use_weights=False,
            )
            self.calibrateSingle(quantiles, targets, i)

    def calibrateSingle(
        self, quantiles: torch.Tensor, targets: torch.Tensor, i: int
    ) -> None:
        b = len(targets)
        mask = targets != 0.0
        alpha = torch.tensor((100 - i) / 100)
        scores = torch.zeros_like(quantiles)

        # calculate score bounds (distance of targets to CI i)
        scores[..., 0] = quantiles[..., 0] - targets
        scores[..., 1] = targets - quantiles[..., 1]

        # get the distance to the closer boundary
        midx = scores.abs().min(-1)[1].unsqueeze(-1)
        scores = torch.gather(scores, -1, midx).squeeze(-1)
        scores[~mask] = float("inf")
        B = mask.sum(0)

        # sort and get desired score quantile
        scores, _ = scores.sort(0, descending=False)
        factor = mask.float().mean(0) * b * 1.01
        idx = (factor * (1 - alpha)).ceil().clamp(max=B - 1).to(torch.int64)
        corrections = torch.gather(scores, dim=0, index=idx.unsqueeze(0)).squeeze(0)

        # store correction
        if quantiles.dim() == 4:
            corrections = corrections.mean(0)
        self.corrections[str(i)] = corrections

    def apply(self, quantiles: torch.Tensor, i: int) -> torch.Tensor:
        Q = self.corrections.get(str(i), None)
        if Q is None:
            print(f"{i}-CI not learned by calibrator")
            return quantiles
        quantiles_c = quantiles.clone()
        quantiles_c[..., 0] -= Q
        quantiles_c[..., 1] += Q
        return quantiles_c

    def insert(self, corrections: dict[str, torch.Tensor]):
        self.corrections = corrections
        self.intervals = [int(k) for k in corrections]

    def save(self, model_id: str, i: int, local: bool = False) -> None:
        suffix = "-local" if local else ""
        fn = Path("outputs", "checkpoints", model_id, f"calibrator_i={i}{suffix}.pt")
        torch.save(self.corrections, fn)
        print(f"Saved calibration values to {fn}.")

    def load(self, model_id: str, i: int) -> None:
        fn = Path("outputs", "checkpoints", model_id, f"calibrator_i={i}.pt")
        corrections = torch.load(fn, weights_only=False)
        self.insert(corrections)
        print(f"Loaded calibration values from {fn}.")


def empiricalCoverage(quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #  how often are the targets actually inside a given credibility interval?
    mask = targets != 0.0
    above = targets >= quantiles[..., 0] - 1e-6  # above lower quantile
    below = targets <= quantiles[..., -1] + 1e-6  # belowe upper quantile
    inside = above * below * mask
    coverage = inside.float().sum(0) / (mask.sum(0) + 1e-12)
    return coverage  # (d,)


def getCoverage(
    model,
    proposed: dict[str, torch.Tensor],
    targets: torch.Tensor,
    intervals: list[int] = CI,
    calibrate: bool = False,
    local: bool = False,
) -> dict[str, torch.Tensor]:
    out = {}
    for i in intervals:
        alpha = (100 - i) / 100
        quantiles = model.quantiles(
            proposed, [alpha / 2, 1 - alpha / 2], calibrate=calibrate, local=local
        )
        ce = empiricalCoverage(quantiles, targets)
        if local:
            ce = ce.mean(0)
        out[str(i)] = ce
    return out


def coverageError(coverage: dict[str, torch.Tensor]) -> torch.Tensor:
    concatenated = torch.cat([v.unsqueeze(0) for _, v in coverage.items()])
    mask = concatenated != 0.0
    nominal = torch.tensor([int(k) for k in coverage.keys()]).unsqueeze(1) / 100
    errors = (concatenated - nominal) * mask
    mean_error = errors.sum(0) / (mask.sum(0) + 1e-12)
    return mean_error


def plotCoverage(
    ax,
    coverage: dict[str, torch.Tensor],
    names,
    linestyle: str = "-",
    lw=2,
    y_name: str = '',
    upper: bool = False,
) -> None:
    nominal = [int(k) for k in coverage.keys()]
    matrix = torch.cat([t.unsqueeze(-1) for _, t in coverage.items()], dim=-1)
    for i, name in enumerate(names):
        color = palette[i]
        coverage_i = matrix[i] * 100.0
        if coverage_i.sum() == 0:
            continue
        ax.plot(
            nominal, coverage_i, label=name, linestyle=linestyle, lw=lw, color=color,
        )
    ax.plot([50, 95], [50, 95], ":", zorder=1, color="grey", label="identity")
    ax.set_xticks(nominal)
    ax.set_yticks(nominal)
    ax.tick_params(axis='both', labelsize=18)

    ax.set_ylabel(y_name, fontsize=26, labelpad=10)
    ax.grid(True)
    if upper:
        ax.legend(fontsize=22, loc="lower right")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelcolor="w", size=1)
    else:
        ax.set_xlabel("Nominal CI", fontsize=26, labelpad=10)

