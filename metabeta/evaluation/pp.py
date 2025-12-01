import torch
from torch import distributions as D
import seaborn as sns
from metabeta.utils import weightedMean


def prepare( 
        ds: dict[str, torch.Tensor],
        proposed: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
    # prepare observed
    X, Z, y = ds["X"], ds["Z"], ds["y"]
    d = X.shape[-1]
    mask = (y != 0).unsqueeze(-1)

    # prepare samples
    samples_g = proposed["global"]["samples"]
    samples_l = proposed["local"]["samples"]
    weights_l = proposed["local"].get("weights", None)
    rfx = weightedMean(samples_l, weights_l).to(X.dtype)
    ffx = samples_g[:, :d].to(X.dtype)
    b = len(ffx)
    sigma_eps = samples_g[:, -1].view(b, 1, 1, -1) + 1e-12

    # construct posterior predictive distribution
    mu_g = torch.einsum("bmnd,bds->bmns", X, ffx)
    mu_l = torch.einsum("bmnq,bmq->bmn", Z, rfx).unsqueeze(-1)

    return dict(mu_g=mu_g, mu_l=mu_l, sigma_eps=sigma_eps, mask=mask)
    
        
def posteriorPredictiveMean(
    ds: dict[str, torch.Tensor],
    proposed: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    mu_g, mu_l, _, mask = prepare(ds, proposed).values()
    pp_mean = mu_g + mu_l
    pp_mean *= mask
    return pp_mean


def posteriorPredictiveDensity(
    ds: dict[str, torch.Tensor],
    proposed: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    mu_g, mu_l, sigma_eps, mask = prepare(ds, proposed).values()
    posterior_predictive = D.Normal(mu_g + mu_l, sigma_eps)
    log_prob = posterior_predictive.log_prob(ds['y'].unsqueeze(-1))
    log_prob *= mask
    return log_prob



def posteriorPredictiveSample(
    ds: dict[str, torch.Tensor],
    proposed: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    
    mu_g, mu_l, sigma_eps, mask = prepare(ds, proposed).values()
    posterior_predictive = D.Normal(mu_g + mu_l, sigma_eps)

    # sample from posterior predictive
    y_rep = posterior_predictive.sample((1,)).squeeze(0)
    y_rep *= mask
    return y_rep


def weightSubset(
    weights: torch.Tensor,  #  (b, s)
    alpha: float = 0.01,
) -> torch.Tensor:
    # get a mask of shape (b, s) that subsets the 1-alpha most likely samples
    b, s = weights.shape
    root = torch.tensor(alpha).expand(b, 1)
    w_sorted, w_idx = weights.sort(dim=-1, descending=False)
    w_inv = torch.argsort(w_idx, -1)
    cdf = torch.cumsum(w_sorted, -1) / s
    cdf = cdf.contiguous()
    root = root.contiguous()
    r_idx = torch.searchsorted(cdf, root).clamp(max=s - 1)
    mask = torch.arange(s).unsqueeze(0) >= r_idx
    mask = mask.gather(-1, w_inv)
    return mask


def plotPosteriorPredictive(
    ax,
    y: torch.Tensor,  # (b,m,n)
    y_rep: torch.Tensor,  # (b,m,n,s)
    is_mask: torch.Tensor | None = None,  # (b,s)
    batch_idx: int = 0,
    n_lines: int = 50,
    title: str = "",
    color: str = "green",
    upper: bool = True,
    show_legend: bool = False,
):
    s = y_rep.shape[-1]

    # prepare data
    y_flat = y[batch_idx].view(-1)
    mask = y_flat != 0
    y_flat = y_flat[mask].numpy()
    y_rep_flat = y_rep[batch_idx].view(-1, s)[mask].numpy()
    if is_mask is not None:
        counts = is_mask.sum(-1)
        _, idx = counts.sort(descending=True)
    else:
        idx = None

    # plot samples with highest IS efficiency
    label = None
    sns.kdeplot(y_flat, color="black", lw=5, label="observed", ax=ax)
    for i in range(n_lines):
        j = idx[i] if idx is not None else i
        y_rep_j = y_rep_flat[..., j]
        if i == n_lines - 1:
            label = "p.p."
        sns.kdeplot(y_rep_j, color=color, alpha=0.15, lw=1.5, label=label, ax=ax)
    sns.kdeplot(
        y_rep_flat.mean(-1),
        linestyle="--",
        color="lightgray",
        lw=3,
        label="p.p. mean",
        ax=ax,
    )
    sns.despine()
    ax.set_yticks(ticks=[])
    ax.set_ylabel("")

    if show_legend:
        ax.legend(fontsize=16, loc="upper right")
    if upper:
        ax.set_xlabel("")
        ax.set_xticks([])
    else:
        ax.set_xlabel("y", labelpad=10, size=26)

