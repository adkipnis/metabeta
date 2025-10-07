import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import pearsonr, gaussian_kde
import numpy as np
import torch

mse = torch.nn.MSELoss()

def posterior(
    target: torch.Tensor,
    proposed: dict[str, torch.Tensor],
    mcmc: dict[str, torch.Tensor],
    names: list[str],
    batch_idx: int = 0,
    **kwargs,
):
    # apply target mask
    mask = target[batch_idx] != 0.0
    target_ = target[batch_idx][mask]
    samples = proposed["samples"]
    samples_ = samples[batch_idx][mask]
    if "weights" in proposed:
        weights_ = proposed["weights"][batch_idx][mask]
    else:
        weights_ = None
    mc_samples = (mcmc["samples"][batch_idx][mask],)
    names_ = names[mask.numpy()]
    d = int(mask.sum())
    w = int(torch.tensor(d).sqrt().ceil())
    fig, axs = plt.subplots(figsize=(8 * w, 5 * w), ncols=w, nrows=w)
    axs = axs.flatten()

    # Plot KDE over a grid
    for i in range(d):
        ax = axs[i]
        ax.set_axisbelow(True)
        # ax.grid(True)
        weights_i = weights_[i] if weights_ is not None else None
        samples_i = samples_[i]
        mc_samples_i = mc_samples[i]

        # grid
        x_left = min(mc_samples_i.min(), samples_i.min())
        x_right = max(mc_samples_i.max(), samples_i.max())
        x_grid = np.linspace(x_left, x_right, 1000)

        # mb kde plot
        label = "MB" if i == 0 else None
        kde = gaussian_kde(samples_i, weights=weights_i)
        ax.plot(x_grid, kde(x_grid), color="darkgreen", label=label, lw=4)

        # mc kde plot
        label = "HMC" if i == 0 else None
        kde_m = gaussian_kde(mc_samples_i)
        ax.plot(x_grid, kde_m(x_grid), color="darkgoldenrod", label=label, lw=4)

        ax.axvline(
            x=target_[i], linestyle="--", linewidth=4, color="black", label="true"
        )
        ax.set_xlabel(names_[i], fontsize=30)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelcolor="w", size=1)
        ax.tick_params(axis="x", labelcolor="w", size=1)
        if i == 0:
            ax.legend(fontsize=26)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for i in range(d, w * w):
        axs[i].set_visible(False)

    fig.tight_layout()


def recovery(
    targets: torch.Tensor,
    names: list[str],
    means: torch.Tensor,
    color: str = "darkgreen",
    alpha: float = 0.3,
    return_stats: bool = True,
) -> None | tuple[float, float]:
    """plot true targets against posterior means for entire batch"""
    assert means.shape[-1] == len(names), "shape mismatch"
    D = len(names)
    if targets.dim() == 3:
        targets = targets.view(-1, D)
        means = means.view(-1, D)
    mask = targets != 0.0
    w = int(torch.tensor(D).sqrt().ceil())

    # init figure
    fig, axs = plt.subplots(figsize=(8 * w, 7 * w), ncols=w, nrows=w)
    if w > 1:
        axs = axs.flatten()

    # init stats
    RMSE = 0.0
    R = 0.0
    denom = D

    # make subplots
    for i in range(D):
        ax = axs[i] if w > 1 else axs
        mask_i = mask[..., i]
        targets_i = targets[mask_i, i]
        mean_i = means[mask_i, i].detach()

        # skip empty target
        if mask_i.sum() == 0:
            axs[i].set_visible(False)
            denom -= 1
            continue

        # compute stats
        r = float(pearsonr(targets_i, mean_i)[0])  # type: ignore
        R += r
        bias = (targets_i - mean_i).mean()
        rmse = mse(targets_i, mean_i).sqrt()
        RMSE += rmse

        # subplot
        ax.set_axisbelow(True)
        ax.grid(True)
        min_val = min(mean_i.min(), targets_i.min()).floor()
        max_val = max(mean_i.max(), targets_i.max()).ceil()
        ax.plot(
            [min_val, max_val], [min_val, max_val], "--", lw=2, zorder=1, color="grey"
        )  # diagline
        ax.scatter(targets_i, mean_i, alpha=alpha, color=color, label=names[i])
        ax.text(
            0.75,
            0.1,
            f"r = {r:.3f}\nBias = {bias.item():.3f}\nRMSE = {rmse.item():.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=16,
            bbox=dict(
                boxstyle="round",
                facecolor=(1, 1, 1, 0.7),
                edgecolor=(0, 0, 0, alpha),
            ),
        )
        ax.set_xlabel("true", fontsize=20)
        ax.set_ylabel("estimated", fontsize=20)
        ax.legend()

    # skip remaining empty subplots
    for i in range(D, w * w):
        axs[i].set_visible(False)
    fig.tight_layout()

    # optionally return average statistics
    if return_stats:
        return RMSE / denom, R / denom


def _recoveryGrouped(
