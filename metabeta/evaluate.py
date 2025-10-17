import argparse
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import time
from collections.abc import Iterable
import torch
from torch import distributions as D
import numpy as np
from scipy.stats import pearsonr
from metabeta.data.dataset import getDataLoader
from metabeta.utils import dsFilename, getConsoleWidth
from metabeta.models.approximators import ApproximatorMFX
from metabeta.evaluation.importance import ImportanceLocal, ImportanceGlobal
from metabeta.evaluation.coverage import getCoverage, plotCalibration, coverageError
from metabeta.evaluation.sbc import getRanks, plotSBC, plotECDF, getWasserstein
from metabeta.evaluation.pp import (
    posteriorPredictiveSample,
    plotPosteriorPredictive,
    weightSubset,
)
from metabeta import plot

CI = [50, 68, 80, 90, 95]
plt.rcParams["figure.dpi"] = 300

###############################################################################


def setup() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--m_tag', type=str, default='r', help='Suffix for model ID (default = '')')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Model seed')
    parser.add_argument('--cores', type=int, default=8, help='Nubmer of processor cores to use (default = 8)')

    # data
    parser.add_argument('--d_tag', type=str, default='r', help='Suffix for data ID (default = '')')
    parser.add_argument('--varied', action='store_true', help='Use data with variable d/q (default = False)')
    parser.add_argument('--semi', action='store_true', help='Use semi-synthetic data (default = True)')
    parser.add_argument('-t', '--fx_type', type=str, default='mfx', help='Type of dataset [ffx, mfx] (default = ffx)')
    parser.add_argument('-d', type=int, default=3, help='Number of fixed effects (with bias, default = 8)')
    parser.add_argument('-q', type=int, default=1, help='Number of random effects (with bias, default = 3)')
    parser.add_argument('-m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('-n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    parser.add_argument('--permute', action='store_false', help='Permute slope variables for uniform learning across heads (default = True)')

    # evaluation
    parser.add_argument('--bs-val', type=int, default=256, help='macro batch size for validation partition (default = 256).')
    parser.add_argument('--bs-test', type=int, default=128, help='macro batch size for test partition (default = 128).')
    parser.add_argument('--bs-mini', type=int, default=32, help='mini batch size (default = 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (Adam, default = 5e-4)')
    parser.add_argument('--standardize', action='store_false', help='Standardize inputs (default = True)')
    parser.add_argument('--importance', action='store_false', help="Do importance sampling (default = True)")
    parser.add_argument('--calibrate', action='store_false', help="Calibrate posterior (default = True)")
    parser.add_argument('--iteration', type=int, default=10, help='Preload model from iteration #p')

    # summary network
    parser.add_argument('--sum_type', type=str, default='set-transformer', help='Summarizer architecture [set-transformer, dual-transformer] (default = set-transformer)')
    parser.add_argument('--sum_blocks', type=int, default=3, help='Number of blocks in summarizer (default = 4)')
    parser.add_argument('--sum_d', type=int, default=128, help='Model dimension (default = 128)')
    parser.add_argument('--sum_ff', type=int, default=128, help='Feedforward dimension (default = 128)')
    parser.add_argument('--sum_depth', type=int, default=1, help='Feedforward layers (default = 1)')
    parser.add_argument('--sum_out', type=int, default=64, help='Summary dimension (default = 64)')
    parser.add_argument('--sum_heads', type=int, default=8, help='Number of heads (poolformer, default = 8)')    
    parser.add_argument('--sum_dropout', type=float, default=0.01, help='Dropout rate (default = 0.01)')
    parser.add_argument('--sum_act', type=str, default='GELU', help='Activation funtction [anything implemented in torch.nn] (default = GELU)')
    parser.add_argument('--sum_sparse', action='store_false', help='Use sparse implementation (poolformer, default = False)')

    # posterior network
    parser.add_argument('--post_type', type=str, default='affine', help='Posterior architecture [affine, spline] (default = affine)')
    parser.add_argument('--flows', type=int, default=3, help='Number of normalizing flow blocks (default = 4)')
    parser.add_argument('--post_ff', type=int, default=128, help='Feedforward dimension (default = 128)')
    parser.add_argument('--post_depth', type=int, default=3, help='Feedforward layers (default = 3)')
    parser.add_argument('--post_dropout', type=float, default=0.01, help='Dropout rate (default = 0.01)')
    parser.add_argument('--post_act', type=str, default='ReLU', help='Activation funtction [anything implemented in torch.nn] (default = ReLU)')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers

def load(model: ApproximatorMFX, model_path, iteration: int) -> None:
    model_path = Path(model_path, model.id)
    fname = Path(model_path, f"checkpoint_i={iteration}.pt")
    print(f"Loading checkpoint from {fname}")
    state = torch.load(fname, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.stats = state["stats"]


def inspect(
    results: dict,
    batch_indices: Iterable[int],
) -> None:
    targets = results["targets"]
    names = results["names"]
    proposed = results["proposed"]
    mcmc = results["mcmc"]

    # visualize some posteriors
    for b in batch_indices:
        plot.posterior(
            target=targets,
            proposed=proposed["global"],
            mcmc=mcmc["global"],
            names=names,
            batch_idx=b,
        )


# -----------------------------------------------------------------------------
# runner


def run(
    model: ApproximatorMFX,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    # model outputs
    with torch.no_grad():
        start = time.perf_counter()
        results = model(batch, sample=True, n=(500, 300))
        end = time.perf_counter()
    print(f"forward pass took {end - start:.2f}s")
    losses = results["loss"]
    proposed = results["proposed"]
    print(f"Mean loss: {losses.mean().item():.4f}")

    # reconstruct varied intercepts for d = 1 and q = 0 from noise estimate
    samples = proposed["global"]["samples"]
    mask = batch["d"] == 1
    means = samples[mask, 0]
    stds = samples[mask, -1]
    intercepts = D.Normal(means, stds).sample((1,)).squeeze(0)
    proposed["global"]["samples"][mask, 0] = intercepts

    # references
    mcmc = None
    if "mcmc_global" in batch:
        mcmc = {}
        mcmc["global"] = {"samples": batch["mcmc_global"]}
        if "mcmc_local" in batch:
            mcmc["local"] = {"samples": batch["mcmc_local"]}

    # outputs
    out = {
        "batch": batch,
        "losses": losses,
        "proposed": proposed,
        "perm": batch["perm"],  # used column permutation
        "unperm": batch["unperm"],  # corresponding unpermutation
        "names": model.names(batch),
        "names_l": model.names(batch, local=True),
        "targets": model.targets(batch),
        "targets_l": model.targets(batch, local=True),
        "mcmc": mcmc,
    }
    return out


# -----------------------------------------------------------------------------
# evaluator


def evaluate(
    model: ApproximatorMFX,
    results: dict,
    importance: bool = False,
    calibrate: bool = False,
    extensive: int = 0,
    iters: int = 10,
) -> dict:
    # unpack
    batch = results["batch"]
    proposed = copy.deepcopy(results["proposed"])
    names = results["names"]
    names_l = results["names_l"]
    targets = results["targets"]
    targets_l = results["targets_l"]

    # importance sampling
    if "flow" in cfg.post_type and importance:
        print("Importance Sampling...")
        start = time.perf_counter()
        for _ in range(iters):
            proposed = ImportanceLocal(batch)(proposed)
            proposed = ImportanceGlobal(batch)(proposed)
        end = time.perf_counter()
        print(f"IS took {end - start:.2f}s")
        sample_efficiency = proposed["global"].get("sample_efficiency")
        if sample_efficiency is not None:
            print(f"Mean IS sample efficiency: {sample_efficiency.mean().item():.2f}")

    # ------------------------------------------------------------------------------
    # recovery MB
    mean, std = model.moments(proposed["global"])
    mean_l, std_l = model.moments(proposed["local"])
    plot.recoveryGrouped(
        targets=[targets[:, : model.d], targets_l, targets[:, model.d :]],
        means=[mean[:, : model.d], mean_l, mean[:, model.d :]],
        names=[names[: model.d], names_l, names[model.d :]],
        titles=["Fixed Effects", "Random Effects", "Variance Parameters"],
    )

    # recovery HMC
    mcmc = results.get("mcmc", None)
    if mcmc is not None:
        m_mean, m_std = model.moments(mcmc["global"])
        m_mean_l, m_std_l = model.moments(mcmc["local"])
        plot.recoveryGrouped(
            targets=[targets[:, : model.d], targets_l, targets[:, model.d :]],
            means=[m_mean[:, : model.d], m_mean_l, m_mean[:, model.d :]],
            names=[names[: model.d], names_l, names[model.d :]],
            titles=["Fixed Effects", "Random Effects", "Variance Parameters"],
            marker="s",
        )

    # ------------------------------------------------------------------------------
    # coverage MB
    coverage = coverage_m = None
    coverage = getCoverage(
        model, proposed["global"], targets, intervals=CI, calibrate=calibrate
    )
    if mcmc is not None:
        coverage_m = getCoverage(
            model, mcmc["global"], targets, intervals=CI, calibrate=False
        )

        fig, axs = plt.subplots(figsize=(7, 7 * 2), ncols=1, nrows=2, dpi=300)
        plotCalibration(axs[0], coverage, names, lw=3, upper=True)
        plotCalibration(axs[1], coverage_m, names, lw=3, upper=False)
        fig.tight_layout()

    # ------------------------------------------------------------------------------
    # SBC histogram MB
    mask_d = targets != 0.0
    ranks = getRanks(targets, proposed["global"], absolute=False, mask_0=True)
    # wd = getWasserstein(ranks, mask_d)
    # print(f"SBC Wasserstein Distance (MB): {wd:.3f}")
    if extensive == 1:
        plotSBC(ranks, mask_d, names, color="darkgreen")

    # SBC histogram MCMC
    if mcmc is not None:
        ranks_m = getRanks(targets, mcmc["global"])
        # wd_m = getWasserstein(ranks_m, mask_d)
        # print(f"SBC Wasserstein Distance (HMC): {wd_m:.3f}")
        if extensive == 1:
            plotSBC(ranks_m, mask_d, names, color="tan")

    # ------------------------------------------------------------------------------
    # ECDF diff MB
    ranks_abs = getRanks(targets, proposed["global"], absolute=True, mask_0=True)
    if extensive == 1:
        plotECDF(
            ranks_abs,
            mask_d,
            names,
            s=proposed["global"]["samples"].shape[-1],
            color="darkgreen",
        )

    # ECDF diff HMC
    if mcmc is not None:
        ranks_abs_m = getRanks(targets, mcmc["global"], absolute=True)
        if extensive == 1:
            plotECDF(
                ranks_abs_m,
                mask_d,
                names,
                s=mcmc["global"]["samples"].shape[-1],
                color="darkgoldenrod",
            )

    # ------------------------------------------------------------------------------
    # posterior predictive plots
    if extensive == 1 and mcmc is not None:
        subset_idx = torch.randperm(1000)[
            :500
        ]  # we need to subsample due to memory demands
        mcmc_sub = {"global": {}, "local": {}}
        mcmc_sub["global"] = {"samples": mcmc["global"]["samples"][..., subset_idx]}
        mcmc_sub["local"] = {"samples": mcmc["local"]["samples"][..., subset_idx]}
        y_rep_mcmc = posteriorPredictiveSample(batch, mcmc_sub)
        y_rep = posteriorPredictiveSample(batch, proposed)
        is_mask = weightSubset(proposed["global"]["weights"][:, 0])

        fig, axs = plt.subplots(figsize=(6 * 2, 5 * 2), ncols=2, nrows=2, dpi=300)
        plotPosteriorPredictive(
            axs[0, 0],
            batch["y"],
            y_rep,
            is_mask,
            batch_idx=0,
            color="green",
            upper=True,
        )
        plotPosteriorPredictive(
            axs[1, 0],
            batch["y"],
            y_rep_mcmc,
            batch_idx=0,
            color="darkgoldenrod",
            upper=False,
        )
        plotPosteriorPredictive(
            axs[0, 1],
            batch["y"],
            y_rep,
            is_mask,
            batch_idx=11,
            color="green",
            upper=True,
            show_legend=True,
        )
        plotPosteriorPredictive(
            axs[1, 1],
            batch["y"],
            y_rep_mcmc,
            batch_idx=11,
            color="darkgoldenrod",
            upper=False,
        )

    return proposed


def quickEval(
    model: ApproximatorMFX,
    results: dict,
    importance: bool = False,
    calibrate: bool = False,
    iters: int = 3,
    table: int = 1,
) -> None:
    # unpack
    batch = results["batch"]
    proposed = copy.deepcopy(results["proposed"])
    targets = results["targets"]
    targets_l = results["targets_l"]
    mcmc = results["mcmc"]

    # importance sampling
    if importance:
        for _ in range(iters):
            proposed = ImportanceLocal(batch)(proposed)
            proposed = ImportanceGlobal(batch)(proposed)

    # recovery MB
    mean, _ = model.moments(proposed["global"])
    rs = np.array(
        [pearsonr(targets[..., i], mean[..., i])[0] for i in range(mean.shape[-1])]
    )
    r_ffx = rs[: model.d].mean()
    r_sigmas = rs[model.d :].mean()

    rmses = (targets - mean).square().mean(0).sqrt()
    rmse_ffx = rmses[: model.d].mean()
    rmse_sigmas = rmses[model.d :].mean()

    mean_l, _ = model.moments(proposed["local"])
    mask_l = targets_l != 0.0
    r_rfx = []
    rmse_rfx = []
    for i in range(mean_l.shape[-1]):
        mask_i = mask_l[..., i]
        targets_i = targets_l[..., i][mask_i]
        means_i = mean_l[..., i][mask_i]
        r_rfx += [pearsonr(targets_i, means_i)[0]]
        rmse_rfx += [(targets_i - means_i).square().mean(0).sqrt()]
    r_rfx = np.mean(r_rfx)
    rmse_rfx = np.mean(rmse_rfx)

    r = (r_ffx + r_sigmas + r_rfx) / 3
    rmse = (rmse_ffx + rmse_sigmas + rmse_rfx) / 3

    # recovery MCMC
    m_mean, _ = model.moments(mcmc["global"])
    m_rs = np.array(
        [pearsonr(targets[..., i], m_mean[..., i])[0] for i in range(m_mean.shape[-1])]
    )
    m_r_ffx = m_rs[: model.d].mean()
    m_r_sigmas = m_rs[model.d :].mean()

    m_rmses = (targets - m_mean).square().mean(0).sqrt()
    m_rmse_ffx = m_rmses[: model.d].mean()
    m_rmse_sigmas = m_rmses[model.d :].mean()

    m_mean_l, _ = model.moments(mcmc["local"])
    mask_l = targets_l != 0.0
    m_r_rfx = []
    m_rmse_rfx = []
    for i in range(mean_l.shape[-1]):
        mask_i = mask_l[..., i]
        targets_i = targets_l[..., i][mask_i]
        m_means_i = m_mean_l[..., i][mask_i]
        m_r_rfx += [pearsonr(targets_i, m_means_i)[0]]
        m_rmse_rfx += [(targets_i - m_means_i).square().mean(0).sqrt()]
    m_r_rfx = np.mean(m_r_rfx)
    m_rmse_rfx = np.mean(m_rmse_rfx)

    m_r = (m_r_ffx + m_r_sigmas + m_r_ffx) / 3
    m_rmse = (m_rmse_ffx + m_rmse_sigmas + m_rmse_rfx) / 3

    # coverage MB
    coverage_g = getCoverage(
        model, proposed["global"], targets, intervals=CI, calibrate=calibrate
    )
    coverage_l = getCoverage(
        model,
        proposed["local"],
        targets_l,
        intervals=CI,
        calibrate=calibrate,
        local=True,
    )
    ce_g = coverageError(coverage_g)
    ce_ffx = ce_g[: model.d].mean()
    ce_sigmas = ce_g[model.d :].mean()
    ce_rfx = coverageError(coverage_l).mean()
    ce = (ce_ffx + ce_sigmas + ce_rfx) / 3

    # coverage MCMC
    m_coverage_g = getCoverage(
        model, mcmc["global"], targets, intervals=CI, calibrate=False
    )
    m_coverage_l = getCoverage(
        model, mcmc["local"], targets_l, intervals=CI, calibrate=False, local=True
    )
    m_ce_g = coverageError(m_coverage_g)
    m_ce_ffx = m_ce_g[: model.d].mean()
    m_ce_sigmas = m_ce_g[model.d :].mean()
    m_ce_rfx = coverageError(m_coverage_l).mean()
    m_ce = (m_ce_ffx + m_ce_sigmas + m_ce_rfx) / 3

    # overleaf: Table 1
    if table == 1:
        overleaf = rf"& {r:.3f} & {rmse:.3f} & {ce:.3f} & {m_r:.3f} & {m_rmse:.3f} & {m_ce.mean():.3f} \\"
        print(overleaf)

    elif table == 2:
        # overleaf: Table 2
        overleaf_ffx = rf"& $\boldsymbol{{\beta}}$ & {r_ffx:.3f} & {rmse_ffx:.3f} & {ce_ffx:.3f} & {m_r_ffx:.3f} & {m_rmse_ffx:.3f} & {m_ce_ffx:.3f} \\"
        overleaf_sig = rf"& $\boldsymbol{{\sigma}}$ & {r_sigmas:.3f} & {rmse_sigmas:.3f} & {ce_sigmas:.3f} & {m_r_sigmas:.3f} & {m_rmse_sigmas:.3f} & {m_ce_sigmas:.3f} \\"
        overleaf_rfx = rf"& $\boldsymbol{{\alpha}}$ & {r_rfx:.3f} & {rmse_rfx:.3f} & {ce_rfx:.3f} & {m_r_rfx:.3f} & {m_rmse_rfx:.3f} & {m_ce_rfx:.3f} \\"
        print(overleaf_ffx)
        print(overleaf_sig)
        print(overleaf_rfx)


# =============================================================================
if __name__ == "__main__":
    # --- setup
    cfg = setup()
    path = Path("outputs", "checkpoints")
    console_width = getConsoleWidth()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cores)
    type_suffix = "-semi" if cfg.semi else ""

    # --- set up model
    summary_dict = {
        "type": cfg.sum_type,
        "d_model": cfg.sum_d,
        "n_blocks": cfg.sum_blocks,
        "d_ff": cfg.sum_ff,
        "depth": cfg.sum_depth,
        "d_output": cfg.sum_out,
        "n_heads": cfg.sum_heads,
        "dropout": cfg.sum_dropout,
        "activation": cfg.sum_act,
        "sparse": cfg.sum_sparse,
    }
    posterior_dict = {
        "type": cfg.post_type,
        "flows": cfg.flows,
        "d_ff": cfg.post_ff,
        "depth": cfg.post_depth,
        "dropout": cfg.post_dropout,
        "activation": cfg.post_act,
    }
    model_dict = {
        "fx_type": cfg.fx_type,
        "seed": cfg.seed,
        "tag": cfg.m_tag,
        "d": cfg.d,
        "q": cfg.q,
    }
    model = ApproximatorMFX.build(
        s_dict=summary_dict,
        p_dict=posterior_dict,
        m_dict=model_dict,
        use_standardization=cfg.standardize,
    )
    model.eval()
    print(f"{'-' * console_width}\nmodel: {model.id}")

    # --- load model and data
    load(model, path, cfg.iteration)
    fn = dsFilename(cfg.fx_type, f"val{type_suffix}",
                    1, cfg.m, cfg.n, cfg.d, cfg.q, cfg.bs_val,
                    varied=cfg.varied,
                    tag=cfg.d_tag,
                    )
    dl_val = getDataLoader(fn, cfg.bs_val, max_d=cfg.d, max_q=cfg.q,
                           permute=False, autopad=True, device="cpu",
                           )
    ds_val = next(iter(dl_val))
    print(f"preloaded model from iteration {cfg.iteration} and test set...\n{'-' * console_width}")

    # --- calibrate on calibration set
    if cfg.calibrate:
        print("\nRunning full sampling from calibration set...")
        results_cal = run(model, ds_val)
        evaluate(model, results_cal, importance=False, calibrate=False, extensive=0)
        evaluate(model, results_cal, importance=cfg.importance, calibrate=False, extensive=1)
        print("Calibrating with conformal prediction...")
        model.calibrator.calibrate(
            model,
            proposed=results_cal["proposed"]["global"], # type: ignore
            targets=results_cal["targets"], # type: ignore
        )
        model.calibrator.save(model.id, cfg.iteration)
        model.calibrator_l.calibrate(
            model,
            proposed=results_cal["proposed"]["local"], # type: ignore
            targets=results_cal["targets_l"], # type: ignore
            local=True,
        )
        model.calibrator_l.save(model.id, cfg.iteration, local=True)

    # --- run on test set
    fn_test = dsFilename(cfg.fx_type, f"test{type_suffix}",
        1, cfg.m, cfg.n, cfg.d, cfg.q, cfg.bs_test,
        varied=cfg.varied, tag=cfg.d_tag,
    )
    dl_test = getDataLoader(
        fn_test, cfg.bs_test, max_d=cfg.d, permute=False, autopad=True, device="cpu"
    )
    ds_test = next(iter(dl_test))

    print("\nRunning and evaluating test set...")
    results_test = run(model, ds_test)
    proposed = evaluate(model, results_test,
                        importance=cfg.importance,
                        calibrate=cfg.calibrate,
                        extensive=2,
                        )
    quickEval(model, results_test,
              importance=cfg.importance,
              calibrate=cfg.calibrate,
              table=1,
              )

    # inspect(results_test, batch_indices=range(10))

    # --- run on varying subsets, separate splits
    ns = ds_test["n"]
    rnv = ds_test["rnv"]
    b = len(ns)

    # set 1: high n
    top_n, idx_size_high = ns.topk(b // 2)
    ds_test_1 = {k: v[idx_size_high] for k, v in ds_test.items()}
    results_test_1 = run(model, ds_test_1)

    # set 2: low n
    idx_size_low = np.setdiff1d(np.arange(b), idx_size_high.numpy())
    ds_test_2 = {k: v[idx_size_low] for k, v in ds_test.items()}
    results_test_2 = run(model, ds_test_2)

    # set 3: high SNR
    bottom_noise, idx_noise_low = (-rnv).topk(b // 2)
    ds_test_3 = {k: v[idx_noise_low] for k, v in ds_test.items()}
    results_test_3 = run(model, ds_test_3)

    # set 4: low SNR
    idx_noise_high = np.setdiff1d(np.arange(b), idx_noise_low.numpy())
    ds_test_4 = {k: v[idx_noise_high] for k, v in ds_test.items()}
    results_test_4 = run(model, ds_test_4)

    quickEval(
        model,
        results_test_1,
        importance=cfg.importance,
        calibrate=cfg.calibrate,
    )
    quickEval(
        model,
        results_test_2,
        importance=cfg.importance,
        calibrate=cfg.calibrate,
    )
    quickEval(
        model,
        results_test_3,
        importance=cfg.importance,
        calibrate=cfg.calibrate,
    )
    quickEval(
        model,
        results_test_4,
        importance=cfg.importance,
        calibrate=cfg.calibrate,
    )
