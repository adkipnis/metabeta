import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde, binom


def getRanks(
    targets: torch.Tensor, proposed: dict, absolute=False, mask_0=False
) -> torch.Tensor:
    # targets (b, d), samples (b, s, d)
    mask = targets != 0
    if mask_0:  # ignore d=1 models
        mask[:, 0] = mask[:, 1:-1].any(1)
    samples = proposed["samples"]
    weights = proposed.get("weights", None)
    targets_ = targets.unsqueeze(-1)
    if absolute:
        samples = samples.abs()
        targets_ = targets_.abs()
    smaller = (samples <= targets_).float()
    if weights is not None:
        ranks = (smaller * weights).sum(-1) / samples.shape[-1]
    else:
        ranks = smaller.mean(-1)
    ranks[~mask] = -1
    return ranks


def plotSBC(
    ranks: torch.Tensor, mask: torch.Tensor | None, names: list, color: str = "darkgreen"
) -> None:
    eps = 0.02
    n = len(names)
    w = int(torch.tensor(n).sqrt().ceil())
    _, axs = plt.subplots(figsize=(8 * w, 6 * w), ncols=w, nrows=w)
    axs = axs.flatten()
    endpoints = binom.interval(0.95, n, 1 / 20)
    mean = n / 20

    for i in range(n):
        ax = axs[i]
        ax.set_axisbelow(True)
        ax.grid(True)
        mask_0 = ranks[:, i] >= 0
        if mask is not None:
            mask_i = mask[:, i] * mask_0
        else:
            mask_i = mask_0
        xx = ranks[mask_i, i].numpy()
        if mask_i.sum() == 0:
            axs[i].set_visible(False)
            continue
        ax.axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.1)
        ax.axhline(mean, color="gray", zorder=0, alpha=0.9, linestyle="--")
        sns.histplot(
            xx,
            kde=True,
            ax=ax,
            binwidth=0.05,
            binrange=(0, 1),
            color=color,
            alpha=0.5,
            stat="density",
            label=names[i],
        )
        ax.set_xlim(0 - eps, 1 + eps)
        ax.set_xlabel("U", fontsize=20)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelcolor="w")
        ax.legend()
    for i in range(n, w * w):
        axs[i].set_visible(False)


def plotSBCsingle(
    ax, ranks: torch.Tensor, n: int, upper: bool = True, color: str = "darkgreen"
) -> None:
    eps = 0.02
    endpoints = binom.interval(0.95, n, 1 / 20)
    mean = n / 20
    ax.axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.1)
    ax.axhline(
        mean,
        color="gray",
        zorder=0,
        alpha=0.9,
        label="theoretical",
        lw=2,
        linestyle="--",
    )
    ax.set_axisbelow(True)
    ax.grid(True)
    mask = ranks >= 0
    xx = ranks[mask].numpy()
    sns.histplot(
        xx,
        kde=True,
        ax=ax,
        binwidth=0.05,
        binrange=(0, 1),
        lw=2,
        color=color,
        alpha=0.5,
        stat="density",
        label="estimated",
    )
    ax.set_xlim(0 - eps, 1 + eps)
    ax.set_ylabel("")

    ax.tick_params(axis="y", labelcolor="w")
    if upper:
        ax.set_title("Calibration", fontsize=30, pad=15)
        ax.legend(fontsize=16, loc="upper right")
        ax.set_xlabel("")
    else:
        ax.set_xlabel("U", labelpad=10, size=26)


def getWasserstein(ranks: torch.Tensor, mask: torch.Tensor, n_points=1000):
