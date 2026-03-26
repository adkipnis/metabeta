"""
Tests whether a trained model actually incorporates prior family information
into its posteriors, rather than ignoring the family encoding.

Approach:
    1. Load a trained model and a batch of test data
    2. Fix the random seed and dataset
    3. For each parameter group (ffx, sigma_rfx, sigma_eps), swap the family
       index while keeping everything else identical
    4. Compare posterior mean and SD under each family
    5. Verify that heavier-tailed families (StudentT, HalfStudentT) produce
       wider posteriors than lighter ones (Normal, HalfNormal)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.config import dataFromYaml, modelFromYaml
from metabeta.utils.io import runName
from metabeta.utils.families import FFX_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.plot import PALETTE, niceify


DIR = Path(__file__).resolve().parent
ROOT = DIR / '..'
CKPT_DIR = ROOT / 'metabeta' / 'outputs' / 'checkpoints'
SEED = 42


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prior family sensitivity experiment.')
    parser.add_argument('--d_tag', type=str, default='toy', help='data config tag')
    parser.add_argument('--m_tag', type=str, default='toy', help='model config tag')
    parser.add_argument('--seed', type=int, default=SEED, help='training seed (for checkpoint lookup)')
    parser.add_argument('--r_tag', type=str, default=None, help='run tag (for checkpoint lookup)')
    parser.add_argument('--n_samples', type=int, default=512, help='posterior samples')
    parser.add_argument('--prefix', type=str, default='latest', help='checkpoint prefix [latest, best]')
    parser.add_argument('--valid', action='store_true', help='use validation set instead of test set')
    parser.add_argument('--delta', action='store_true', help='plot delta variance instead of absolute')
    return parser.parse_args()
# fmt: on


def loadData(cfg: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Load a single batch from test (default) or validation set."""
    data_cfg_path = ROOT / 'metabeta' / 'simulation' / 'configs' / f'{cfg.d_tag}.yaml'
    partition = 'valid' if cfg.valid else 'test'
    data_fname = dataFromYaml(data_cfg_path, partition)
    data_path = ROOT / 'metabeta' / 'outputs' / 'data' / data_fname
    assert data_path.exists(), f'{data_path} not found'
    dl = Dataloader(data_path)
    return next(iter(dl))


def loadModel(cfg: argparse.Namespace, d: int, q: int) -> Approximator:
    """Load model from latest (or best) checkpoint."""
    model_cfg_path = ROOT / 'metabeta' / 'models' / 'configs' / f'{cfg.m_tag}.yaml'
    model_cfg = modelFromYaml(model_cfg_path, d, q)
    model = Approximator(model_cfg)

    # find checkpoint
    run = runName(vars(cfg))
    ckpt_path = CKPT_DIR / run / f'{cfg.prefix}.pt'
    assert ckpt_path.exists(), f'checkpoint not found: {ckpt_path}'
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model_state'])
    epoch = payload.get('epoch', '?')
    print(f'  loaded {cfg.prefix} checkpoint (epoch {epoch}) from {ckpt_path}')

    model.eval()
    return model


def resetRng(model: Approximator) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    model.posterior_g.base_dist.base.rng = np.random.default_rng(SEED)   # type: ignore
    model.posterior_l.base_dist.base.rng = np.random.default_rng(SEED)   # type: ignore


@torch.inference_mode()
def getPosteriorStats(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run inference and return posterior mean and SD per parameter group (rescaled)."""
    resetRng(model)
    proposal = model.estimate(batch, n_samples=n_samples)
    proposal.rescale(batch['sd_y'])
    return {
        'ffx': {
            'mean': proposal.ffx.mean(dim=-2),  # (b, d)
            'sd': proposal.ffx.std(dim=-2),  # (b, d)
        },
        'sigma_rfx': {
            'mean': proposal.sigma_rfx.mean(dim=-2),  # (b, q)
            'sd': proposal.sigma_rfx.std(dim=-2),  # (b, q)
        },
        'sigma_eps': {
            'mean': proposal.sigma_eps.mean(dim=-1),  # (b,)
            'sd': proposal.sigma_eps.std(dim=-1),  # (b,)
        },
    }


def setBatchFamily(
    batch: dict[str, torch.Tensor],
    key: str,
    value: int,
) -> dict[str, torch.Tensor]:
    """Clone batch and set all family indices for one group to a fixed value."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    out[key] = torch.full_like(out[key], value)
    return out


# ---- Step 1: check that posterior changes when family changes ----


def checkPosteriorChanges(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
) -> None:
    """For each parameter group, verify that swapping the family index
    changes the posterior mean and SD."""
    print('\n--- Step 1: Does the posterior change when the family changes? ---')

    groups = [
        ('family_ffx', FFX_FAMILIES, 'ffx'),
        ('family_sigma_rfx', SIGMA_FAMILIES, 'sigma_rfx'),
        ('family_sigma_eps', SIGMA_FAMILIES, 'sigma_eps'),
    ]

    for family_key, families, param_key in groups:
        print(f'\n  [{param_key}] comparing {len(families)} families:')
        stats_per_family = {}
        for i, name in enumerate(families):
            b = setBatchFamily(batch, family_key, i)
            stats_per_family[name] = getPosteriorStats(model, b, n_samples)

        # compare all pairs
        names = list(stats_per_family.keys())
        for j in range(len(names)):
            for k in range(j + 1, len(names)):
                s_j = stats_per_family[names[j]][param_key]
                s_k = stats_per_family[names[k]][param_key]
                mean_diff = (s_j['mean'] - s_k['mean']).abs().mean().item()
                sd_diff = (s_j['sd'] - s_k['sd']).abs().mean().item()
                mean_changed = not torch.allclose(s_j['mean'], s_k['mean'], atol=1e-6)
                sd_changed = not torch.allclose(s_j['sd'], s_k['sd'], atol=1e-6)
                status = 'PASS' if (mean_changed and sd_changed) else 'FAIL'
                print(
                    f'    {names[j]} vs {names[k]}: '
                    f'mean diff={mean_diff:.6f}, sd diff={sd_diff:.6f} [{status}]'
                )


# ---- Step 2: check direction (heavier tails → wider posteriors) ----

# tail weight ordering per group
TAIL_ORDER = [
    ('family_ffx', 'ffx', [('normal', 0), ('student', 1)]),
    (
        'family_sigma_rfx',
        'sigma_rfx',
        [('halfnormal', 0), ('halfstudent', 1), ('exponential', 2)],
    ),
    (
        'family_sigma_eps',
        'sigma_eps',
        [('halfnormal', 0), ('halfstudent', 1), ('exponential', 2)],
    ),
]


def collectTailStats(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Collect per-dataset posterior SDs for each family and parameter group.

    Returns:
        {param_key: {family_name: (b,) or (b, d/q) tensor of SDs}}
    """
    results = {}
    for family_key, param_key, ordered_families in TAIL_ORDER:
        results[param_key] = {}
        for name, idx in ordered_families:
            b = setBatchFamily(batch, family_key, idx)
            stats = getPosteriorStats(model, b, n_samples)
            results[param_key][name] = stats[param_key]['sd']
    return results


def checkTailDirection(
    tail_stats: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Print monotonic ordering check from pre-collected stats."""
    print('\n--- Step 2: Do heavier-tailed families widen the posterior? ---')

    for _, param_key, ordered_families in TAIL_ORDER:
        names = [n for n, _ in ordered_families]
        print(f'\n  [{param_key}] tail ordering: {" < ".join(names)}')
        sds = []
        for name in names:
            sd = tail_stats[param_key][name].mean().item()
            sds.append((name, sd))
            print(f'    {name}: SD={sd:.4f}')

        for j in range(len(sds) - 1):
            wider = sds[j + 1][1] > sds[j][1]
            ratio = sds[j + 1][1] / (sds[j][1] + 1e-12)
            status = 'PASS' if wider else 'FAIL'
            print(f'    {sds[j][0]} -> {sds[j + 1][0]}: ratio={ratio:.3f} [{status}]')


# ---- Visualization ----


PARAM_COLORS = {
    'ffx': PALETTE[0],
    'sigma_rfx': PALETTE[2],
    'sigma_eps': PALETTE[4],
}
PARAM_LABELS = {
    'ffx': r'$\beta$',
    'sigma_rfx': r'$\sigma_0$',
    'sigma_eps': r'$\sigma_\epsilon$',
}

# panels: (title, light_key, heavy_key, light_label, heavy_label, param_keys to overlay)
MICRO_PANELS = [
    ('Normal vs. Student-t', 'normal', 'student', 'Normal', 'Student-t', ['ffx']),
    (
        'Half-Normal vs. Half-Student-t',
        'halfnormal',
        'halfstudent',
        'Half-Normal',
        'Half-Student-t',
        ['sigma_rfx', 'sigma_eps'],
    ),
    (
        'Half-Student-t vs. Exponential',
        'halfstudent',
        'exponential',
        'Half-Student-t',
        'Exponential',
        ['sigma_rfx', 'sigma_eps'],
    ),
]

SIZE_RANGE = (15.0, 120.0)


def _pointSizes(
    n: torch.Tensor,
    n_param_dims: int,
) -> np.ndarray:
    """Map total observation counts to marker sizes, repeating per parameter dim."""
    n_np = n.float().numpy()
    n_min, n_max = n_np.min(), n_np.max()
    if n_max > n_min:
        sizes = np.interp(n_np, (n_min, n_max), SIZE_RANGE)
    else:
        sizes = np.full_like(n_np, np.mean(SIZE_RANGE), dtype=float)
    return np.repeat(sizes, n_param_dims)


def plotMicro(
    tail_stats: dict[str, dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    out_path: Path | None = None,
    delta: bool = False,
) -> None:
    """Three panels: x = posterior variance (lighter family).
    y = posterior variance (heavier family) if delta=False, else delta variance.
    Colors distinguish parameter groups; point size reflects total n."""
    n = batch['n']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    for i, (ax, (title, light, heavy, light_lbl, heavy_lbl, param_keys)) in enumerate(
        zip(axes, MICRO_PANELS)
    ):
        all_x, all_y, all_s, all_c = [], [], [], []

        for pk in param_keys:
            sd_light = tail_stats[pk][light]
            sd_heavy = tail_stats[pk][heavy]
            var_light = (sd_light**2).flatten().numpy()
            var_heavy = (sd_heavy**2).flatten().numpy()
            n_dims = sd_light.shape[-1] if sd_light.dim() > 1 else 1
            sizes = _pointSizes(n, n_dims)
            all_x.append(var_light)
            all_y.append(var_heavy - var_light if delta else var_heavy)
            all_s.append(sizes)
            all_c.extend([PARAM_COLORS[pk]] * len(var_light))

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        s = np.concatenate(all_s)

        ax.scatter(x, y, s=s, c=all_c, alpha=0.45, edgecolors='none')
        if delta:
            ax.axhline(0, ls='--', color='grey', lw=1, alpha=0.6)
        else:
            lo = min(x.min(), y.min()) * 0.95
            hi = max(x.max(), y.max()) * 1.05
            if 'ffx' in param_keys:
                lo, hi = 0.0, 3.8
            ax.plot([lo, hi], [lo, hi], '--', color='grey', lw=1, alpha=0.6)
            step = 0.5
            ticks = np.arange(np.floor(lo / step) * step, hi + step, step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        # stats
        var_heavy_raw = np.concatenate(
            [(tail_stats[pk][heavy] ** 2).flatten().numpy() for pk in param_keys]
        )
        var_ratio = var_heavy_raw / (x + 1e-12)
        frac_above = (var_heavy_raw > x).mean()
        ylabel = rf'$\Delta$ Var({heavy_lbl})' if delta else f'Var({heavy_lbl})'
        niceify(
            ax,
            {
                'title': title,
                'title_fs': 22,
                'xlabel': f'Var({light_lbl})',
                'xlabel_fs': 20,
                'ylabel': ylabel,
                'ylabel_fs': 20,
                'show_legend': False,
                'stats': {'Var-ratio': var_ratio.mean(), '% above': frac_above * 100},
                'stats_suffix': '',
                'stats_loc_x': 0.65,
                'stats_loc_y': 0.05,
            },
        )

    # color legend: right of the rightmost plot
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=PARAM_COLORS[pk],
            markersize=16,
            label=PARAM_LABELS[pk],
        )
        for pk in ('ffx', 'sigma_rfx', 'sigma_eps')
    ]
    # size legend based on n
    n_np = n.float().numpy()
    n_levels = np.unique(np.round(np.quantile(n_np, [0.0, 0.5, 1.0])).astype(int))
    n_min, n_max = n_np.min(), n_np.max()
    size_handles = []
    for nl in n_levels:
        sl = float(np.interp(float(nl), (n_min, n_max), SIZE_RANGE)) if n_max > n_min else 60.0
        size_handles.append(plt.scatter([], [], s=sl, c='grey', alpha=0.5, label=f'{nl}'))

    leg1 = axes[-1].legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.75),
        fontsize=20,
        title='',
        title_fontsize=20,
    )
    axes[-1].add_artist(leg1)
    axes[-1].legend(
        handles=size_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.30),
        fontsize=20,
        title='n (total)',
        title_fontsize=20,
    )

    fig.suptitle('Posterior variance under prior family', fontsize=26, x=0.42, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    if out_path is not None:
        fig.savefig(out_path / 'prior_families_micro.png', bbox_inches='tight', dpi=150)
    plt.show()


if __name__ == '__main__':
    cfg = setup()
    print('Prior Family Sensitivity Experiment')

    # load data and model
    batch = loadData(cfg)
    d = int(batch['mask_d'].shape[-1])
    q = int(batch['mask_q'].shape[-1])
    model = loadModel(cfg, d, q)
    print(f'  d={d}, q={q}, batch_size={batch["y"].shape[0]}')

    # run checks
    # checkPosteriorChanges(model, batch, cfg.n_samples)
    tail_stats = collectTailStats(model, batch, cfg.n_samples)
    checkTailDirection(tail_stats)

    # visualize
    out_path = DIR / 'plots'
    out_path.mkdir(exist_ok=True)
    plotMicro(tail_stats, batch, out_path, delta=cfg.delta)

    print('\nDone.')
