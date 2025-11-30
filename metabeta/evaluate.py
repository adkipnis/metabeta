import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
from scipy.stats import pearsonr
from metabeta.data.dataset import getDataLoader
from metabeta.utils import setDevice, dsFilename, getConsoleWidth
from metabeta.models.approximators import ApproximatorMFX
from metabeta.evaluation.importance import ImportanceLocal, ImportanceGlobal
from metabeta.evaluation.coverage import getCoverage, plotCalibration
from metabeta.evaluation.sbc import getRanks, plotSBC, plotECDF
from metabeta.evaluation.pp import (
    posteriorPredictiveDensity,
    posteriorPredictiveSample,
    plotPosteriorPredictive,
    weightSubset,
)
from metabeta import plot

plt.rcParams['figure.dpi'] = 300

###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()
    
    # misc
    parser.add_argument('-s', '--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--device', type=str, default='mps', help='device to use [cpu, cuda, mps], (default = mps)')
    parser.add_argument('--cores', type=int, default=8, help='nubmer of processor cores to use (default = 8)')
    
    # loading
    parser.add_argument('--d_tag', type=str, default='all', help='suffix for data ID (default = '')')
    parser.add_argument('--m_tag', type=str, default='all', help='suffix for model ID (default = '')')
    parser.add_argument('--c_tag', type=str, default='config', help='name of model config file (default = "config")')
    parser.add_argument('-l', '--load', type=int, default=10, help='load model from iteration #l')
    
    # evaluation
    parser.add_argument('--bs_val', type=int, default=256, help='number of regression datasets in validation set (default = 256)')
    parser.add_argument('--bs-test', type=int, default=256, help='number of regression datasets in test set  (default = 256).')
    parser.add_argument('--bs_mini', type=int, default=16, help='umber of regression datasets per minibatch (default = 16)')
    parser.add_argument('--importance', action='store_false', help='Do importance sampling (default = True)')
    parser.add_argument('--calibrate', action='store_false', help='Calibrate posterior (default = True)')
    
    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers

def load(model: ApproximatorMFX, model_path, iteration: int) -> None:
    model_path = Path(model_path, model.id)
    fname = Path(model_path, f'checkpoint_i={iteration}.pt')
    print(f'Loading checkpoint from {fname}')
    state = torch.load(fname, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.stats = state['stats']


def run(
    model: ApproximatorMFX,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    # model outputs
    with torch.no_grad():
        start = time.perf_counter()
        results = model(batch, sample=True, n=(500, 300))
        end = time.perf_counter()
    print(f'forward pass took {end - start:.2f}s')
    losses = results['loss']
    proposed = results['proposed']
    print(f'Mean loss: {losses.mean().item():.4f}')

    # references
    nuts = None
    if 'nuts_global' in batch:
        nuts = {}
        nuts['global'] = {'samples': batch['nuts_global']}
        if 'nuts_local' in batch:
            nuts['local'] = {'samples': batch['nuts_local']}

    # outputs
    out = {
        'batch': batch,
        'losses': losses,
        'proposed': proposed,
        'perm': batch['perm'],  # used column permutation
        'unperm': batch['unperm'],  # corresponding unpermutation
        'names': model.names(batch),
        'names_l': model.names(batch, local=True),
        'targets': model.targets(batch),
        'targets_l': model.targets(batch, local=True),
        'nuts': nuts,
    }
    return out


def estimate(model: ApproximatorMFX,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    # model outputs
    with torch.no_grad():
        start = time.perf_counter()
        proposed = model.estimate(batch, n=(500, 300))
        end = time.perf_counter()
    print(f'forward pass took {end - start:.2f}s')

    # references
    nuts = None
    if 'nuts_global' in batch:
        nuts = {}
        nuts['global'] = {'samples': batch['nuts_global']}
        if 'nuts_local' in batch:
            nuts['local'] = {'samples': batch['nuts_local']}

    # outputs
    out = {
        'batch': batch,
        'proposed': proposed,
        'nuts': nuts,
    }
    return out


# -----------------------------------------------------------------------------
# refinements

def importanceSampling(results: dict, iters: int = 2) -> None:
    batch = results['batch']
    proposed = results['proposed']
    start = time.perf_counter()
    for _ in range(iters):
        proposed = ImportanceLocal(batch)(proposed)
        proposed = ImportanceGlobal(batch)(proposed)
    end = time.perf_counter()
    print(f'IS took {end - start:.2f}s')
    sample_efficiency = proposed['global'].get('sample_efficiency')
    if sample_efficiency is not None:
        print(f'Mean IS sample efficiency: {sample_efficiency.mean().item():.2f}')
    
    
def calibrate(model: ApproximatorMFX, results: dict) -> None:
    model.calibrator.calibrate(model,
        proposed=results['proposed']['global'], # type: ignore
        targets=results['targets'], # type: ignore
    )
    model.calibrator.save(model.id, cfg.load)
    model.calibrator_l.calibrate(
        model,
        proposed=results['proposed']['local'], # type: ignore
        targets=results['targets_l'], # type: ignore
        local=True,
    )
    model.calibrator_l.save(model.id, cfg.load, local=True)


# -----------------------------------------------------------------------------
# evaluators

def recovery(model: ApproximatorMFX, results: dict) -> None:
    proposed = results['proposed']
    targets = results['targets']
    targets_l = results['targets_l']
    names = results['names']
    names_l = results['names_l']
    nuts = results.get('nuts')
    
    # metabeta
    mean, std = model.moments(proposed['global'])
    mean_l, std_l = model.moments(proposed['local'])
    plot.recoveryGrouped(
        targets=[targets[:, : model.d], targets_l, targets[:, model.d :]],
        means=[mean[:, : model.d], mean_l, mean[:, model.d :]],
        names=[names[: model.d], names_l, names[model.d :]],
        titles=['Fixed Effects', 'Random Effects', 'Variance Parameters'],
    )

    # NUTS
    if nuts is not None:
        m_mean, m_std = model.moments(nuts['global'])
        m_mean_l, m_std_l = model.moments(nuts['local'])
        plot.recoveryGrouped(
            targets=[targets[:, : model.d], targets_l, targets[:, model.d :]],
            means=[m_mean[:, : model.d], m_mean_l, m_mean[:, model.d :]],
            names=[names[: model.d], names_l, names[model.d :]],
            titles=['Fixed Effects', 'Random Effects', 'Variance Parameters'],
            marker='s',
        )


def inSampleLikelihood(results: dict):
    # in-sample posterior predictive likelihood
    batch = results['batch']
    proposed = results['proposed']
    nuts = results.get('nuts')
    
    # metabeta
    y_log_prob = posteriorPredictiveDensity(batch, proposed)
    pp_nll = -y_log_prob.sum(dim=(1,2)).mean(-1)
    results['pp_nnl'] = {'mb': pp_nll}
    
    if nuts is not None:
        # sub-sample due to memory constraints
        s = nuts['global']['samples'].shape[-1]
        subset_idx = torch.randperm(s)[:1000] # we need to subsample due to memory demands
        nuts_sub = {'global': {}, 'local': {}}
        nuts_sub['global'] = {'samples': nuts['global']['samples'][..., subset_idx]}
        nuts_sub['local'] = {'samples': nuts['local']['samples'][..., subset_idx]}
        
        # evaluate
        y_log_prob_m = posteriorPredictiveDensity(batch, nuts_sub)
        pp_nll_m = -y_log_prob_m.sum(dim=(1,2)).mean(-1)
        results['pp_nnl'] = {'nuts': pp_nll_m}
        
        # plot
        mask_mb = (pp_nll < 1e4)
        mask_mcmc = (pp_nll_m < 1e4)
        mask_pp = mask_mcmc * mask_mb
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(pp_nll[mask_pp], pp_nll_m[mask_pp], 'o')
        r = float(pearsonr(pp_nll[mask_pp], pp_nll_m[mask_pp])[0])
        print(f'Cor of pp log likelihoods {r:.3f}')
        
            
# # Comparison of mean posterior y
# y_rep_mean = posteriorPredictiveMean(batch, proposed).mean(-1)
# pp_se = (batch['y'] - y_rep_mean).square().view(b, -1)
# pp_rmse = (pp_se.sum(-1) / batch['mask_n'].view(b, -1).sum(-1)).sqrt()
# print(pp_rmse.mean())
# if nuts is not None:
#     y_rep_mean_m = posteriorPredictiveMean(batch, nuts).mean(-1)
#     pp_se_m = (batch['y'] - y_rep_mean_m).square().view(b, -1)
#     pp_rmse_m = (pp_se_m.sum(-1) / batch['mask_n'].view(b, -1).sum(-1)).sqrt()
#     plt.plot(pp_rmse, pp_rmse_m, 'o')
#     print(pp_rmse_m.mean())


def posteriorPredictive(results: dict, index: int = 0):
    batch = results['batch']
    proposed = results['proposed']
    nuts = results.get('nuts')
    n_plots = 2 if nuts is not None else 1
    fig, axs = plt.subplots(figsize=(6, 5 * n_plots), ncols=1, nrows=n_plots, dpi=300)
    if n_plots == 1:
        axs = [axs]
    
    # metabeta
    y_rep = posteriorPredictiveSample(batch, proposed)
    is_mask = weightSubset(proposed['global']['weights'][:, 0])
    plotPosteriorPredictive(
        axs[0],
        batch['y'],
        y_rep,
        is_mask,
        batch_idx=0,
        color='green',
        upper=True,
    )
    
    # NUTS
    if nuts is not None:
        # sub-sample due to memory constraints
        s = nuts['global']['samples'].shape[-1]
        subset_idx = torch.randperm(s)[:1000] # we need to subsample due to memory demands
        nuts_sub = {'global': {}, 'local': {}}
        nuts_sub['global'] = {'samples': nuts['global']['samples'][..., subset_idx]}
        nuts_sub['local'] = {'samples': nuts['local']['samples'][..., subset_idx]}
        
        # sample
        y_rep_nuts = posteriorPredictiveSample(batch, nuts_sub)
        plotPosteriorPredictive(
            axs[1, 0],
            batch['y'],
            y_rep_nuts,
            batch_idx=0,
            color='darkgoldenrod',
            upper=False,
        )


def coverage(model: ApproximatorMFX, results: dict, use_calibrated: bool = True) -> None:
    proposed = results['proposed']
    targets = results['targets']
    names = results['names']
    nuts = results.get('nuts')
    n_plots = 2 if nuts is not None else 1
    fig, axs = plt.subplots(figsize=(7, 7 * n_plots), ncols=1, nrows=n_plots, dpi=300)
    if n_plots == 1:
        axs = [axs]
    
    # metabeta
    coverage = getCoverage(
        model, proposed['global'], targets, calibrate=use_calibrated
    )
    plotCalibration(axs[0], coverage, names, lw=3, upper=True)
    
    # NUTS
    if nuts is not None:
        coverage_m = getCoverage(
            model, nuts['global'], targets, calibrate=False
        )
        plotCalibration(axs[1], coverage_m, names, lw=3, upper=False)
    fig.tight_layout()


def sbc(results: dict):
    proposed = results['proposed']
    targets = results['targets']
    mask_d = (targets != 0)
    names = results['names']
    nuts = results.get('nuts')
    
    # --- SBC histogram 
    # metabeta
    ranks = getRanks(targets, proposed['global'], absolute=False, mask_0=True)
    plotSBC(ranks, mask_d, names, color='darkgreen')
    # wd = getWasserstein(ranks, mask_d)

    # NUTS
    if nuts is not None:
        ranks_m = getRanks(targets, nuts['global'])
        plotSBC(ranks_m, mask_d, names, color='tan')
        # wd_m = getWasserstein(ranks_m, mask_d)

    # --- SBC ECDF
    # metabeta
    ranks_abs = getRanks(targets, proposed['global'], absolute=True, mask_0=True)
    plotECDF(ranks_abs, mask_d, names,
             s=proposed['global']['samples'].shape[-1],
             color='darkgreen')

    # NUTS
    if nuts is not None:
        ranks_abs_m = getRanks(targets, nuts['global'], absolute=True)
        plotECDF(ranks_abs_m, mask_d, names, 
                 s=nuts['global']['samples'].shape[-1],
                 color='darkgoldenrod')



# =============================================================================
if __name__ == '__main__':
    # --- setup config
    cfg = setup()
    path = Path('outputs', 'checkpoints')
    console_width = getConsoleWidth()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cores)


    # --- setup and load model
    with open(Path('models', f'{cfg.c_tag}.yaml'), 'r') as f:
        model_cfg = yaml.safe_load(f)
        model_cfg['general']['seed'] = cfg.seed
        model_cfg['general']['tag'] = cfg.m_tag
    model = ApproximatorMFX.build(model_cfg)
    model.eval()
    load(model, path, cfg.load)
    print(f'{'-' * console_width}\nmodel: {model.id}')


    # --- load validation set
    fn_val = dsFilename('mfx', 'val', 1,
                        model_cfg['general']['m'], model_cfg['general']['n'],
                        model_cfg['general']['d'], model_cfg['general']['q'], 
                        size=cfg.bs_val, tag=cfg.d_tag)
    dl_val = getDataLoader(fn_val, cfg.bs_val,
                           max_d=model_cfg['general']['d'],
                           max_q=model_cfg['general']['q'],
                           permute=False, autopad=True, device='cpu')
    ds_val = next(iter(dl_val))
    
    
    # # --- load test set
    # fn_test = dsFilename('mfx', 'test', 1,
    #                      model_cfg['general']['m'], model_cfg['general']['n'],
    #                      model_cfg['general']['d'], model_cfg['general']['q'], 
    #                      size=cfg.bs_test, tag=cfg.d_tag)
    # dl_test = getDataLoader(fn_test, cfg.bs_test,
    #                         max_d=model_cfg['general']['d'],
    #                         max_q=model_cfg['general']['q'],
    #                         permute=False, autopad=True, device='cpu')
    # ds_test = next(iter(dl_test))

    # --- calibrate on calibration set
    if cfg.calibrate:
        print('\nRunning full sampling from calibration set...')
        results_cal = run(model, ds_val)
        evaluate(model, results_cal, importance=cfg.importance, calibrate=False, extensive=0)
        print('Calibrating with conformal prediction...')
        model.calibrator.calibrate(
            model,
            proposed=results_cal['proposed']['global'], # type: ignore
            targets=results_cal['targets'], # type: ignore
        )
        model.calibrator.save(model.id, cfg.load)
        model.calibrator_l.calibrate(
            model,
            proposed=results_cal['proposed']['local'], # type: ignore
            targets=results_cal['targets_l'], # type: ignore
            local=True,
        )
        model.calibrator_l.save(model.id, cfg.load, local=True)

    # # --- run on semi-synthetic test set
    # fn_test = dsFilename(cfg.fx_type, 'test',
    #     1, cfg.m, cfg.n, cfg.d, cfg.q, cfg.bs_test,
    #     # varied=cfg.varied,
    #     tag=cfg.d_tag,
    # )
    # dl_test = getDataLoader(
    #     fn_test, cfg.bs_test, max_d=cfg.d, permute=False, autopad=True, device='cpu'
    # )
    # ds_test = next(iter(dl_test))

    # print('\nRunning and evaluating test set...')
    # results_test = run(model, ds_test)
    # proposed = evaluate(model, results_test,
    #                     importance=False,#cfg.importance,
    #                     calibrate=cfg.calibrate,
    #                     extensive=2,
    #                     )
    
    # quickEval(model, results_test,
    #           importance=cfg.importance,
    #           calibrate=cfg.calibrate,
    #           table=1,
    #           )

    # inspect(results_test, batch_indices=range(10))

    # # --- run on varying subsets, separate splits
    # ns = ds_test['n']
    # rnv = ds_test['rnv']
    # b = len(ns)

    # # set 1: high n
    # top_n, idx_size_high = ns.topk(b // 2)
    # ds_test_1 = {k: v[idx_size_high] for k, v in ds_test.items()}
    # results_test_1 = run(model, ds_test_1)

    # # set 2: low n
    # idx_size_low = np.setdiff1d(np.arange(b), idx_size_high.numpy())
    # ds_test_2 = {k: v[idx_size_low] for k, v in ds_test.items()}
    # results_test_2 = run(model, ds_test_2)

    # # set 3: high SNR
    # bottom_noise, idx_noise_low = (-rnv).topk(b // 2)
    # ds_test_3 = {k: v[idx_noise_low] for k, v in ds_test.items()}
    # results_test_3 = run(model, ds_test_3)

    # # set 4: low SNR
    # idx_noise_high = np.setdiff1d(np.arange(b), idx_noise_low.numpy())
    # ds_test_4 = {k: v[idx_noise_high] for k, v in ds_test.items()}
    # results_test_4 = run(model, ds_test_4)

    # quickEval(
    #     model,
    #     results_test_1,
    #     importance=cfg.importance,
    #     calibrate=cfg.calibrate,
    # )
    # quickEval(
    #     model,
    #     results_test_2,
    #     importance=cfg.importance,
    #     calibrate=cfg.calibrate,
    # )
    # quickEval(
    #     model,
    #     results_test_3,
    #     importance=cfg.importance,
    #     calibrate=cfg.calibrate,
    # )
    # quickEval(
    #     model,
    #     results_test_4,
    #     importance=cfg.importance,
    #     calibrate=cfg.calibrate,
    # )
    
    # # --- run on sub-sampled test set
    # cfg.bs_test = 32
    # fn_sub = dsFilename(cfg.fx_type, 'test-sub',
    #     1, cfg.m, cfg.n, cfg.d, cfg.q, cfg.bs_test,
    #     tag=cfg.d_tag,
    # )
    # dl_sub = getDataLoader(
    #     fn_sub, cfg.bs_test, max_d=cfg.d, max_q=cfg.q, permute=False, autopad=True, device='cpu'
    # )
    # ds_sub = next(iter(dl_sub))
    
    # results_sub = estimate(model, ds_sub)
    
    # mean_ffx = results_sub['proposed']['global']['samples'][:, : model.d].mean(-1)
    # mean_ffx_m = results_sub['nuts']['global']['samples'][:, : model.d].mean(-1)
    # plt.plot(mean_ffx[:, 2], mean_ffx_m[:, 2], 'o')
