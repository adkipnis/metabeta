import argparse
import yaml
import time
from pathlib import Path
from functools import reduce

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch

from metabeta import plot
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
    parser.add_argument('--importance', action='store_false', help='do importance sampling (default = True)')
    parser.add_argument('--calibrate', action='store_false', help='calibrate posterior (default = True)')
    parser.add_argument('--sub', action='store_true', help='evaluate sub-sampled real data (default = False)')
    
    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers

def load(model: ApproximatorMFX, model_path, iteration: int) -> None:
    model_path = Path(model_path, model.id)
    fname = Path(model_path, f'checkpoint_i={iteration}.pt')
    print(f'Loading checkpoint from {fname}')
    state = torch.load(fname, weights_only=False, map_location=torch.device(cfg.device))
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
        t0 = time.perf_counter()
        results = model(batch, sample=True, n=(500, 300))
        t1 = time.perf_counter()
    duration = t1 - t0
    print(f'forward pass took {duration:.2f}s')
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
        'duration': duration,
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
        t0 = time.perf_counter()
        results = model.estimate(batch, n=(500, 300))
        t1 = time.perf_counter()
    duration = t1 - t0
    print(f'forward pass took {duration:.2f}s')
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
        'duration': duration,
        'proposed': proposed,
        'summary': summary,
        'nuts': nuts,
        'advi': advi,
    }
    return out


def subsample(nuts: dict, n: int = 1000) -> dict:
    # sub-sample due to memory constraints
    s = nuts['global']['samples'].shape[-1]
    subset_idx = torch.randperm(s)[:n] # we need to subsample due to memory demands
    nuts_sub = {'global': {}, 'local': {}}
    nuts_sub['global'] = {'samples': nuts['global']['samples'][..., subset_idx]}
    nuts_sub['local'] = {'samples': nuts['local']['samples'][..., subset_idx]}
    return nuts_sub


def removeDivs(ds: dict[str, torch.Tensor], limit: int = 20) -> dict[str, torch.Tensor]:
    # remove data with poor convergence
    few_divs = ds['nuts_divergences'].sum(-1) < limit
    ds = {k: v[few_divs] for k, v in ds.items()}
    return ds


def quarter(ds: dict[str, torch.Tensor]):
    ns = ds['n']
    rsq = ds['r_squared']
    b = len(ns)

    # ---- high vs. low n ----
    _, idx_size_high = ns.topk(b // 2)
    mask_size_low = ~torch.isin(torch.arange(b, device=ns.device), idx_size_high)
    idx_size_low = mask_size_low.nonzero(as_tuple=True)[0]

    ds_N = {k: v[idx_size_high] for k, v in ds.items()}
    ds_n = {k: v[idx_size_low]  for k, v in ds.items()}

    # ---- high vs. low snr ----
    _, idx_noise_low = rsq.topk(b // 2)
    mask_noise_high = ~torch.isin(torch.arange(b, device=rsq.device), idx_noise_low)
    idx_noise_high = mask_noise_high.nonzero(as_tuple=True)[0]

    ds_R = {k: v[idx_noise_low]  for k, v in ds.items()}
    ds_r = {k: v[idx_noise_high] for k, v in ds.items()}

    return ds_N, ds_n, ds_R, ds_r


def subrow(out: dict, name: str):
    fx_types = ['ffx', 'rfx', 'sigmas']
    nll = out[name]['nll']
    r = sum(out[name][fx]['r'] for fx in fx_types) / len(fx_types)
    rmse = sum(out[name][fx]['rmse'] for fx in fx_types) / len(fx_types)
    ce = sum(out[name][fx]['ce'] for fx in fx_types) / len(fx_types)
    t = out[name]['duration']['single']
    return f'{nll:.1f} & {r:.3f} & {rmse:.3f} & {ce:.3f} & {t:.1f}'

def tablerow(out: dict) -> str:
    mb = subrow(out, 'metabeta')
    nuts = subrow(out, 'nuts')
    advi = subrow(out, 'advi')
    mb_name = '\\texttt{metabeta}'
    nuts_name = '\\texttt{HMC}'
    advi_name = '\\texttt{VI}'
    return f'{cfg.d} & {cfg.q} & {mb_name} & {mb}\n  &   & {nuts_name} & {nuts}\n  &   & {advi_name} & {advi}'
    

# -----------------------------------------------------------------------------
# refinements

def importanceSampling(results: dict, iters: int = 2, constrain: bool = True) -> None:
    batch = results['batch']
    proposed = results['proposed']
    if 'weights' in proposed['global']:
        del proposed['global']['weights']
    if 'weights' in proposed['local']:
        del proposed['local']['weights']
    t0 = time.perf_counter()
    IL = ImportanceLocal(batch, constrain=constrain)
    IG = ImportanceGlobal(batch, constrain=constrain)
    for _ in range(iters):
        proposed = IL(proposed)
        proposed = IG(proposed)
    t1 = time.perf_counter()
    results['is_duration'] = float(t1-t0)
    results['is_sample_eff_g'] = float(proposed['global']['sample_efficiency'].median())
    results['is_sample_eff_l'] = float(proposed['local']['sample_efficiency'].median())
    
    
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
# visual evaluators

def recovery(model: ApproximatorMFX, results: dict) -> None:
    proposed = results['proposed']
    targets = results['targets']
    targets_l = results['targets_l']
    names = results['names']
    names_l = results['names_l']
    nuts = results.get('nuts')
    n_plots = 2 if nuts is not None else 1
    fig, axs = plt.subplots(figsize=(6.1 * 3, 6.1 * n_plots), ncols=3, nrows=n_plots, dpi=300)
    if n_plots == 1:
        axs = [axs]
        
    # metabeta
    mean, std = model.moments(proposed['global'])
    mean_l, std_l = model.moments(proposed['local'])
    plot.recoveryGrouped(
        axs[0],
        targets=[targets[:, : model.d], targets[:, model.d :], targets_l],
        means=[mean[:, : model.d], mean[:, model.d :], mean_l],
        names=[names[: model.d], names[model.d :], names_l],
        limits=[(-3.2,3.2), (-0.05, 1.35), (-3.2,3.2)],
        titles=['Fixed Effects', 'Variance Parameters', 'Random Effects'],
        y_name='Estimate', #'metabeta',
        upper=True,
    )
    
    # NUTS
    if nuts is not None:
        m_mean, m_std = model.moments(nuts['global'])
        m_mean_l, m_std_l = model.moments(nuts['local'])
        plot.recoveryGrouped(
            axs[1],
            targets=[targets[:, : model.d], targets[:, model.d :], targets_l],
            means=[m_mean[:, : model.d], m_mean[:, model.d :], m_mean_l],
            names=[names[: model.d], names[model.d :], names_l],
            limits=[(-3.2,3.2), (-0.05, 1.35), (-3.2,3.2)],
            titles=['Fixed Effects', 'Variance Parameters', 'Random Effects'],
            marker='s',
            y_name='Estimate',#'HMC',
            upper=False,
        )
        
    fig.tight_layout()
    fig.savefig(Path(results_path, f'recovery_{data_type}.png'), dpi=300, bbox_inches='tight')
    

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
    if 'weights' in proposed['global']:
        is_mask = weightSubset(proposed['global']['weights'][:, 0])
    else:
        is_mask = None
    plotPosteriorPredictive(
        axs[0],
        batch['y'],
        y_rep,
        is_mask=is_mask,
        batch_idx=0,
        color='green',
        upper=True,
        show_legend=True,
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
            show_legend=False,
        )
    fig.tight_layout()
    fig.savefig(Path(results_path, f'pp_{data_type}.png'), dpi=300, bbox_inches='tight')


def coverage(model: ApproximatorMFX, results: dict, use_calibrated: bool = True) -> None:
    proposed = results['proposed']
    targets = results['targets']
    names = results['names']
    nuts = results.get('nuts')
    n_plots = 2 if nuts is not None else 1
    fig, axs = plt.subplots(figsize=(6.5, 6 * n_plots), ncols=1, nrows=n_plots, dpi=300)
    if n_plots == 1:
        axs = [axs]
    
    # metabeta
    coverage = getCoverage(
        model, proposed['global'], targets, calibrate=use_calibrated
    )
    plotCoverage(axs[0], coverage, names, lw=3,
                 y_name='Empirical CI',#'metabeta',
                 upper=True)
    
    # NUTS
    if nuts is not None:
        coverage_m = getCoverage(
            model, nuts['global'], targets, calibrate=False
        )
        plotCoverage(axs[1], coverage_m, names, lw=3,
                     y_name='Empirical CI',#'HMC',
                     upper=False)
    fig.tight_layout()
    fig.savefig(Path(results_path, f'coverage_{data_type}.png'), dpi=300, bbox_inches='tight')


def hist(results: dict):
    proposed = results['proposed']
    targets = results['targets']
    mask_d = (targets != 0)
    names = results['names']
    nuts = results.get('nuts')
    n_plots = 2 if nuts is not None else 1
    fig, axs = plt.subplots(figsize=(6.5, 6 * n_plots), ncols=1, nrows=n_plots, dpi=300)
    if n_plots == 1:
        axs = [axs]
    
    # metabeta
    ranks = getRanks(targets, proposed['global'], absolute=False, mask_0=True)
    plotSBC(ranks, mask_d, names, color='darkgreen')
    # wd = getWasserstein(ranks, mask_d)

    # NUTS
    if nuts is not None:
        ranks_m = getRanks(targets, nuts['global'])
        plotSBC(ranks_m, mask_d, names, color='tan')
        # wd_m = getWasserstein(ranks_m, mask_d)    


def ecdf(results: dict):
    proposed = results['proposed']
    targets = results['targets']
    mask_d = (targets != 0)
    names = results['names']
    nuts = results.get('nuts')
    n_plots = 2 if nuts is not None else 1
    fig, axs = plt.subplots(figsize=(6.5, 6 * n_plots), ncols=1, nrows=n_plots, dpi=300)
    if n_plots == 1:
        axs = [axs]
    
    # metabeta
    ranks_abs = getRanks(targets, proposed['global'], absolute=True, mask_0=True)
    plotECDF(axs[0], ranks_abs, mask_d, names,
             s=proposed['global']['samples'].shape[-1],
             color='darkgreen', upper=True)

    # NUTS
    if nuts is not None:
        ranks_abs_m = getRanks(targets, nuts['global'], absolute=True)
        plotECDF(axs[1], ranks_abs_m, mask_d, names, 
                 s=nuts['global']['samples'].shape[-1],
                 color='darkgoldenrod', upper=False)
        
    fig.tight_layout()
    fig.savefig(Path(results_path, f'sbc_{data_type}.png'), dpi=300, bbox_inches='tight')


# -----------------------------------------------------------------------------
# numeric evaluators

def _separateRecovery(targets_g: torch.Tensor, targets_l: torch.Tensor,
                      means_g: torch.Tensor, means_l: torch.Tensor,
                      pivot: int) -> dict[str, float]:
    out = {}
    out['ffx'] = _quickRecovery(targets_g[:, :pivot], means_g[:, :pivot])
    out['sigmas'] = _quickRecovery(targets_g[:, pivot:], means_g[:, pivot:])
    out['rfx'] = _quickRecovery(targets=targets_l, means=means_l)
    return out    
    

def quickRecovery(model: ApproximatorMFX, results: dict) -> dict:
    proposed = results['proposed']
    targets_g = results['targets']
    targets_l = results['targets_l']
    nuts = results.get('nuts')
    advi = results.get('advi')
    out = {}    
    
    # metabeta
    means_g_mb, _ = model.moments(proposed['global'])
    means_l_mb, _ = model.moments(proposed['local'])
    out['metabeta'] = _separateRecovery(targets_g, targets_l, means_g_mb, means_l_mb, model.d)
    
    # nuts
    means_g_nuts, _ = model.moments(nuts['global'])
    means_l_nuts, _ = model.moments(nuts['local'])
    out['nuts'] = _separateRecovery(targets_g, targets_l, means_g_nuts, means_l_nuts, model.d)
    
    # advi
    means_g_advi, _ = model.moments(advi['global'])
    means_l_advi, _ = model.moments(advi['local'])
    out['advi'] = _separateRecovery(targets_g, targets_l, means_g_advi, means_l_advi, model.d)
    
    # agreement
    # rmse, r = _quickRecovery(means_g_mb, means_g_nuts)
    
    return out


def _separateCoverage(targets_g: torch.Tensor, targets_l: torch.Tensor,
                      proposed: dict[str, torch.Tensor],
                      pivot: int, use_calibrated: bool = True) -> dict[str, float]:
    out = {'ffx':{}, 'sigmas':{}, 'rfx':{}}
    coverage_g = getCoverage(model, proposed['global'], targets_g,
                             calibrate=use_calibrated, local=False)
    coverage_l = getCoverage(model, proposed['local'], targets_l,
                             calibrate=use_calibrated, local=True)
    ce_g = coverageError(coverage_g)
    ce_l = coverageError(coverage_l)    
    out['ffx']['ce'] = float(ce_g[:pivot].mean())
    out['sigmas']['ce'] = float(ce_g[pivot:].mean())
    out['rfx']['ce'] = float(ce_l.mean())
    return out 


def quickCoverage(model: ApproximatorMFX, results: dict, use_calibrated: bool = True) -> dict:
    proposed = results['proposed']
    targets_g = results['targets']
    targets_l = results['targets_l']
    nuts = results.get('nuts')
    advi = results.get('advi')
    out = {} 
    
    out['metabeta'] = _separateCoverage(targets_g, targets_l, proposed, model.d)
    out['nuts'] = _separateCoverage(targets_g, targets_l, nuts, model.d)
    out['advi'] = _separateCoverage(targets_g, targets_l, advi, model.d)
    return out
    

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

def inSampleLikelihood(results: dict, limit: float = 2000.) -> dict:
    # in-sample posterior predictive likelihood
    batch = results['batch']
    proposed = results['proposed']
    nuts = results.get('nuts')
    advi = results.get('advi')
    out = {'metabeta':{}, 'nuts':{}, 'advi':{}}
    
    # metabeta
    y_log_prob = posteriorPredictiveDensity(batch, proposed)
    nll_mb = -y_log_prob.sum(dim=(1,2)).mean(-1)
    mask_mb = (nll_mb < limit)

    # nuts
    nuts_sub = subsample(nuts) # due to memory constraints
    y_log_prob = posteriorPredictiveDensity(batch, nuts_sub)
    nll_nuts = -y_log_prob.sum(dim=(1,2)).mean(-1)
    mask_nuts = (nll_nuts < limit)
    
    # advi
    advi_sub = subsample(advi)
    y_log_prob = posteriorPredictiveDensity(batch, advi_sub)
    nll_advi = -y_log_prob.sum(dim=(1,2)).mean(-1)
    mask_advi = (nll_advi < limit)
    
    # joint
    mask = mask_mb * mask_nuts * mask_advi
    nll_mb = nll_mb[mask]
    nll_nuts = nll_nuts[mask]
    nll_advi = nll_advi[mask]
    r_mn = float(pearsonr(nll_mb, nll_nuts)[0])
    r_ma = float(pearsonr(nll_mb, nll_advi)[0])
    r_na = float(pearsonr(nll_nuts, nll_advi)[0])
    print(f'NLL Correlation: NUTS={r_mn:.2f}, ADVI={r_ma:.3f}, N/A={r_na:.3f}')
    
    # medians
    out['metabeta']['nll'] = float(nll_mb.median())
    out['nuts']['nll'] = float(nll_nuts.median())
    out['advi']['nll'] = float(nll_advi.median())
    return out


def runtimes(results: dict) -> dict:
    out = {'metabeta':{}, 'nuts':{}, 'advi':{}}
    b = len(results_test['batch']['X'])
    out['metabeta']['duration'] = {
        'single': results['duration'] / b,
        'batch': results['duration']}
    out['nuts']['duration'] = {
        'single': float(results['nuts']['duration'].median()),
        'batch': float(results['nuts']['duration'].sum())}
    out['advi']['duration'] = {
        'single': float(results['advi']['duration'].median()),
        'batch': float(results['advi']['duration'].sum())}
    return out
      
      
def diagnostics(results: dict) -> dict:
    out = {'metabeta':{}, 'nuts':{}, 'advi':{}}
    
    # importance sampling
    out['metabeta']['is_duration'] = results['is_duration']
    out['metabeta']['is_sample_eff_g'] = results['is_sample_eff_g']
    out['metabeta']['is_sample_eff_l'] = results['is_sample_eff_l']
    
    # nuts
    out['nuts']['divergences'] = results['nuts']['divergences'].float().mean().item()
    rhats = results['nuts']['rhat']
    out['nuts']['rhat'] = float(rhats.sum() / (rhats != 0).sum())
    ess = results['nuts']['ess']
    out['nuts']['ess'] = float(ess.sum() / (ess != 0).sum())
    
    # advi
    ess = results['advi']['ess']
    out['advi']['ess'] = float(ess.sum() / (ess != 0).sum())
    return out


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
    
    # --- results path
    results_path = Path('outputs', 'results', model.id)
    results_path.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------------------------------
    # validation set
    data_type = 'val'
    
    # --- load data
    fn_val = dsFilename('mfx', data_type, 1,
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
        
    # # --- performance plots
    # recovery(model, results_val)
    # posteriorPredictive(results_val)
    # coverage(model, results_val, use_calibrated=cfg.calibrate)
    # ecdf(results_val)
    
    
    # -------------------------------------------------------------------------
    # test set (with MCMC estimates)
    data_type = 'test'
        
    # --- load data
    fn_test = dsFilename('mfx', data_type, 1,
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
    
    # --- performance plots
    recovery(model, results_test)
    posteriorPredictive(results_test)
    coverage(model, results_test, use_calibrated=cfg.calibrate)
    ecdf(results_test)
    
    # --- performance measures
    outputs = []
    outputs += [quickRecovery(model, results_test)]
    outputs += [quickCoverage(model, results_test)]
    outputs += [inSampleLikelihood(results_test)]
    outputs += [runtimes(results_test)]
    outputs += [diagnostics(results_test)]
    out = reduce(deepMerge, outputs, {})
    with open(Path(results_path, data_type+'.yaml'), 'w') as f:
        yaml.dump(out, f, sort_keys=False)
    
    
    # -------------------------------------------------------------------------
    # sub-sampled real data (with MCMC estimates)
    data_type = 'test-sub'
    
    # --- load data
    fn_sub = dsFilename('mfx', data_type, 1,
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
        
    # --- performance measures
    outputs = []
    outputs += [inSampleLikelihood(results_sub)]
    outputs += [runtimes(results_sub)]
    outputs += [diagnostics(results_sub)]
    out = reduce(deepMerge, outputs, {})
    with open(Path(results_path, data_type+'.yaml'), 'w') as f:
        yaml.dump(out, f, sort_keys=False)
    
