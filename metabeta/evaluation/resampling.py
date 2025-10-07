import torch
from torch import distributions as D
from sklearn.preprocessing import PowerTransformer


def powersample(
    samples: torch.Tensor, t: int = 1000, method: str = "yeo-johnson"
) -> tuple[torch.Tensor, torch.Tensor]:
    # perform power transform transform to normal space separately per d
    # resample t times from normal space and backtransform to initial space
    if method == "yeo-johnson":
        # only use this with unconstrained data
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        ld = yeojohnsonLogDet
    elif method == "box-cox":
        # only use this with constrained data
        assert (samples <= 0).sum() == 0, "samples must be all positive for box cox"
        pt = PowerTransformer(method="box-cox", standardize=True)
        ld = boxcoxLogDet
    else:
        raise ValueError(
            f"method must be in [yeo-johnson, box-cox] but {method} was supplied"
        )

    b, d, s = samples.shape
    dtype = samples.dtype
    samples_ = samples.permute(0, 2, 1).reshape(b * s, d)
    transformed = pt.fit_transform(samples_)
    transformed = (
        torch.tensor(transformed, dtype=dtype).reshape(b, s, d).permute(0, 2, 1)
    )

    # parameterize new normal and sample from it
    mean = transformed.mean(-1, keepdim=True)
    std = transformed.std(-1, keepdim=True)
    mask = (std == 0.0).squeeze(-1)
    std += 1e-3
    eps = torch.randn((b, d, t))
    eps = (eps - eps.mean(-1, keepdim=True)) / eps.std(-1, keepdim=True)
    new_samples = mean + eps * std
    new_samples[mask] = mean[mask]  # keep constant columns constant
    log_prob = D.Normal(mean, std).log_prob(new_samples)
    if method == "box-cox":  # prevent nans
        new_samples = new_samples.clamp(min=transformed.min())

    # retransform new samples to original space
    new_samples_ = new_samples.permute(0, 2, 1).reshape(b * t, d)
    untransformed = pt.inverse_transform(new_samples_)
    untransformed = torch.tensor(untransformed, dtype=dtype)

    # calculate the log_probs with change of variables
    lambdas = torch.tensor(pt.lambdas_, dtype=dtype).unsqueeze(-1)
    log_det = ld(untransformed.permute(1, 0), lambdas).permute(1, 0)

    # outputs
    resamples = untransformed.reshape(b, t, d).permute(0, 2, 1)
    log_det = log_det.reshape(b, t, d).permute(0, 2, 1)

    # check nans
    sums = resamples.isnan().sum()
    if sums > 0:
        print(f"{sums} nans in resamples")
        resamples = torch.where(resamples.isnan(), 0, resamples)
    return resamples, log_prob + log_det


def boxcoxLogDet(y: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
