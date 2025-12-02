import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
from functools import reduce
from scipy.stats import pearsonr
from metabeta.data.dataset import getDataLoader
from metabeta.utils import setDevice, dsFilename, getConsoleWidth, deepMerge
from metabeta.utils import quickRecovery as _quickRecovery
from metabeta.models.approximators import ApproximatorMFX
from metabeta.evaluation.importance import ImportanceLocal, ImportanceGlobal
from metabeta.evaluation.coverage import getCoverage, plotCoverage, coverageError
from metabeta.evaluation.sbc import getRanks, plotSBC, plotECDF
from metabeta.evaluation.pp import (
    posteriorPredictiveDensity,
    posteriorPredictiveSample,
    plotPosteriorPredictive,
    weightSubset,
)
from metabeta import plot

###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()
    
    # misc
    parser.add_argument('-s', '--seed', type=int, default=42, help='model seed (default = 42)')
    parser.add_argument('--device', type=str, default='cuda', help='device to use [cpu, cuda, mps], (default = mps)')
    parser.add_argument('--cores', type=int, default=8, help='number of processor cores to use (default = 8)')
    
    # model
    parser.add_argument('-d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes)')
    parser.add_argument('-q', type=int, default=1, help='Maximum number of random effects (intercept + slopes)')
    parser.add_argument('--d_tag', type=str, default='all', help='suffix for data ID (default = '')')
    parser.add_argument('--m_tag', type=str, default='all', help='suffix for model ID (default = '')')
    parser.add_argument('--c_tag', type=str, default='default', help='name of model config file')
    parser.add_argument('-l', '--load', type=int, default=50, help='load model from iteration #l')
    
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


def extract(batch: dict[str, torch.Tensor], name: str) -> dict[str, torch.Tensor] | None:
    out = None
    if f'{name}_global' in batch:
        out =  {'duration': batch[f'{name}_duration'],
                'ess': batch[f'{name}_ess']}
        out['global'] = {'samples': batch[f'{name}_global'].cpu()}
        if f'{name}_local' in batch:
            out['local'] = {'samples': batch[f'{name}_local'].cpu()}
        if f'{name}_divergences' in batch:
            out['divergences'] = batch[f'{name}_divergences']
            out['rhat'] = batch[f'{name}_rhat']
    return out


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
    summary = results['summary']
    print(f'Mean loss: {losses.mean().item():.4f}')

    # references
    nuts = extract(batch, 'nuts')
    advi = extract(batch, 'advi')
    
    # put batch to cpu
    batch = {k: v.cpu() for k,v in batch.items()}
    
    # outputs
    out = {
        'batch': batch,
        'losses': losses,
        'proposed': proposed,
        'summary': summary,
        'names': model.names(batch),
        'names_l': model.names(batch, local=True),
        'targets': model.targets(batch).cpu(),
        'targets_l': model.targets(batch, local=True),
        'nuts': nuts,
        'advi': advi,
    }
    return out


def estimate(model: ApproximatorMFX,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    # model outputs
    with torch.no_grad():
        start = time.perf_counter()
        results = model.estimate(batch, n=(500, 300))
        end = time.perf_counter()
    print(f'forward pass took {end - start:.2f}s')
    proposed = results['proposed']
    summary = results['summary']

    # references
    nuts = extract(batch, 'nuts')
    advi = extract(batch, 'advi')
    
    # put batch to cpu
    batch = {k: v.cpu() for k,v in batch.items()}
    
    # outputs
    out = {
        'batch': batch,
        'proposed': proposed,
        'summary': summary,
        'nuts': nuts,
        'advi': advi,
    }
    return out


# -----------------------------------------------------------------------------
# refinements

def importanceSampling(results: dict, iters: int = 2, constrain: bool = True) -> None:
    batch = results['batch']
    proposed = results['proposed']
    if 'weights' in proposed['global']:
        del proposed['global']['weights']
    if 'weights' in proposed['local']:
        del proposed['local']['weights']
    start = time.perf_counter()
    IL = ImportanceLocal(batch, constrain=constrain)
    IG = ImportanceGlobal(batch, constrain=constrain)
    for _ in range(iters):
        proposed = IL(proposed)
        proposed = IG(proposed)
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

def subsample(nuts: dict, n: int = 1000) -> dict:
    # sub-sample due to memory constraints
    s = nuts['global']['samples'].shape[-1]
    subset_idx = torch.randperm(s)[:n] # we need to subsample due to memory demands
    nuts_sub = {'global': {}, 'local': {}}
    nuts_sub['global'] = {'samples': nuts['global']['samples'][..., subset_idx]}
    nuts_sub['local'] = {'samples': nuts['local']['samples'][..., subset_idx]}
    return nuts_sub
    

def inSampleLikelihood(results: dict, limit: float = 2000.):
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
        nuts_sub = subsample(nuts)
        
        # evaluate
        y_log_prob_m = posteriorPredictiveDensity(batch, nuts_sub)
        pp_nll_m = -y_log_prob_m.sum(dim=(1,2)).mean(-1)
        results['pp_nnl'] = {'nuts': pp_nll_m}
        
        # plot
        mask_mb = (pp_nll < limit)
        mask_mcmc = (pp_nll_m < limit)
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
        nuts_sub = subsample(nuts)
        
        # sample
        y_rep_nuts = posteriorPredictiveSample(batch, nuts_sub)
        plotPosteriorPredictive(
            axs[1],
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
    device = setDevice(cfg.device)


    # --- setup and load model
    with open(Path('models', 'configs', f'{cfg.c_tag}.yaml'), 'r') as f:
        model_cfg = yaml.safe_load(f)
        model_cfg['general']['seed'] = cfg.seed
        model_cfg['general']['tag'] = cfg.m_tag
        model_cfg['general']['d'] = cfg.d
        model_cfg['general']['q'] = cfg.q
    model = ApproximatorMFX.build(model_cfg).to(device)
    model.eval()
    load(model, path, cfg.load)
    print(f'{'-' * console_width}\nmodel: {model.id}')


    # -------------------------------------------------------------------------
    # validation set
    
    # --- load data
    fn_val = dsFilename('mfx', 'val', 1,
                        model_cfg['general']['m'], model_cfg['general']['n'],
                        cfg.d, cfg.q, 
                        size=cfg.bs_val, tag=cfg.d_tag)
    dl_val = getDataLoader(fn_val, cfg.bs_val,
                           max_d=cfg.d,
                           max_q=cfg.q,
                           permute=False, autopad=True, device=device)
    ds_val = next(iter(dl_val))
    
    # --- run and refine model
    print('\nInference on validation set...')
    results_val = run(model, ds_val)
    if cfg.importance:
        importanceSampling(results_val)
    if cfg.calibrate:
        calibrate(model, results_val)
        
    # --- evaluate performance
    recovery(model, results_val)
    inSampleLikelihood(results_val)
    posteriorPredictive(results_val)
    coverage(model, results_val, use_calibrated=cfg.calibrate)
    sbc(results_val)
    
    
    # -------------------------------------------------------------------------
    # test set (with MCMC estimates)
        
    # --- load data
    fn_test = dsFilename('mfx', 'test', 1,
                         model_cfg['general']['m'], model_cfg['general']['n'],
                         cfg.d, cfg.q, 
                         size=cfg.bs_test, tag=cfg.d_tag)
    dl_test = getDataLoader(fn_test, cfg.bs_test,
                            max_d=cfg.d,
                            max_q=cfg.q,
                            permute=False, autopad=True, device=device)
    ds_test = next(iter(dl_test))

    
    # --- run and refine model
    print('\nInference on test set...')
    results_test = run(model, ds_test)
    if cfg.importance:
        importanceSampling(results_test)
    
    # --- evaluate performance
    recovery(model, results_test)
    inSampleLikelihood(results_test)
    posteriorPredictive(results_test)
    coverage(model, results_test, use_calibrated=cfg.calibrate)
    sbc(results_test)
    

    # -------------------------------------------------------------------------
    # sub-sampled real data (with MCMC estimates)
    
    # --- load data
    fn_sub = dsFilename('mfx', 'test-sub', 1,
                        model_cfg['general']['m'], model_cfg['general']['n'],
                        cfg.d, cfg.q, 
                        size=cfg.bs_test, tag=cfg.d_tag)
    dl_sub = getDataLoader(fn_sub, cfg.bs_test,
                           max_d=cfg.d,
                           max_q=cfg.q,
                           permute=False, autopad=True, device=device)
    ds_sub = next(iter(dl_sub))
    
    # --- run and refine model
    print('\nInference on sub-sampled real data...')
    results_sub = estimate(model, ds_sub)
    if cfg.importance:
        importanceSampling(results_sub)
        
    # --- evaluate performance
    inSampleLikelihood(results_test)
    # mean_ffx = results_sub['proposed']['global']['samples'][:, : model.d].mean(-1)
    # mean_ffx_m = results_sub['nuts']['global']['samples'][:, : model.d].mean(-1)
    # plt.plot(mean_ffx[:, 2], mean_ffx_m[:, 2], 'o')

