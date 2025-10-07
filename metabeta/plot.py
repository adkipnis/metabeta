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
