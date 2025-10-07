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
    ax: Axes,
    targets: torch.Tensor,
    names: list[str],
    means: torch.Tensor,
    stds: torch.Tensor | None = None,
    title: str = "",
    marker: str = "o",
    colors: list[np.ndarray] | None = None,
    alpha: float = 0.2,
    show_y: bool = True,
) -> None | tuple[float, float]:
    """plot true targets against posterior means for entire batch"""
    # get sizes
    assert means.shape[-1] == len(names), "shape mismatch"
    if colors is not None:
        assert len(colors) >= len(names), "not enough colors provided"
    D = len(names)
    if targets.dim() == 3:
        targets = targets.view(-1, D)
        means = means.view(-1, D)
        stds = stds.view(-1, D) if stds is not None else None
    mask = targets != 0.0

    # init figure
    ax.set_title(title, fontsize=30, pad=15)
    ax.set_axisbelow(True)
    ax.grid(True)
    min_val = float(min(means.min(), targets.min()).floor())
    max_val = float(max(means.max(), targets.max()).ceil())
    addon = 4 if min_val < 0 else 1
    limits = (min_val - addon, max_val + addon)
    ax.set_xlim(limits, auto=False)
    ax.set_ylim(limits, auto=False)
    ax.plot(
        [min_val, max_val], [min_val, max_val], "--", lw=2, zorder=1, color="grey"
    )  # diagline

    # init stats
    RMSE = torch.tensor(0.0)
    R = torch.tensor(0.0)
    Bias = torch.tensor(0.0)
    denom = D

    # overlay plots
    for i in range(D):
        mask_i = mask[..., i]
        targets_i = targets[mask_i, i]
        mean_i = means[mask_i, i].detach()
        # skip empty target
        if mask_i.sum() == 0:
            denom -= 1
            continue

        # compute stats
        r = float(pearsonr(targets_i, mean_i)[0])  # type: ignore
        R += r
        bias = (targets_i - mean_i).mean()
        Bias += bias
        rmse = mse(targets_i, mean_i).sqrt()
        RMSE += rmse

        # subplot
        if colors is not None:
            ax.scatter(
                targets_i,
                mean_i,
                marker=marker,
                alpha=alpha,
                color=colors[i],
                label=names[i],
            )
        else:
            ax.scatter(targets_i, mean_i, marker=marker, alpha=alpha, label=names[i])
    # add stats
    rmse = RMSE / denom
    bias = Bias / denom
    r = R / denom
    ax.text(
        0.7,
        0.1,
        f"r = {r.item():.3f}\nBias = {bias.item():.3f}\nRMSE = {rmse.item():.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=22,
        bbox=dict(
            boxstyle="round",
            facecolor=(1, 1, 1, 0.7),
            edgecolor=(0, 0, 0, alpha),
        ),
    )
    ax.set_xlabel("true", fontsize=26, labelpad=10)
    if show_y:
        ax.set_ylabel("estimated", fontsize=26, labelpad=10)
    ax.legend(fontsize=22, markerscale=2.5, loc="upper left")


def recoveryGrouped(
    targets: list[torch.Tensor],
    names: list[list[str]],
    means: list[torch.Tensor],
    titles: list[str] = [],
    marker: str = "o",
    alpha: float = 0.2,
) -> None | tuple[float, float]:
    N = len(names)
    assert N > 0, "no names provided"
    fig, axs = plt.subplots(figsize=(7 * N, 7), ncols=N, nrows=1, dpi=300)
    i = 0
    for _targets, _names, _means, title, ax in zip(targets, names, means, titles, axs):
        _recoveryGrouped(
            ax,
            _targets,
            _names,
            _means,
            title=title,
            marker=marker,
            alpha=alpha,
            show_y=(i == 0),
        )
        i += 1
    fig.tight_layout()


# compare posterior intervals with mcmc
