"""
Tests whether a trained model actually incorporates prior family information
into its posteriors, rather than ignoring the family encoding.

Approach:
    1. Load trained models and test data for each evaluation config
    2. For each parameter group (ffx, sigma_rfx, sigma_eps), swap the family
       index while keeping everything else identical
    3. Compare posterior SD under each family
    4. Verify that heavier-tailed families (StudentT, HalfStudentT) produce
       wider posteriors than lighter ones (Normal, HalfNormal)
    5. Aggregate results across configs and plot

Usage (from experiments/):
    uv run python prior_families.py
    uv run python prior_families.py --configs mid-n-mixed large-n-mixed
    uv run python prior_families.py --delta
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from metabeta.models.approximator import Approximator
from metabeta.utils.config import assimilateConfig, loadDataConfig, modelFromYaml
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.io import datasetFilename, runName
from metabeta.utils.families import FFX_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.plot import PALETTE, niceify


DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
CKPT_DIR = METABETA / 'outputs' / 'checkpoints'

DEFAULT_CONFIGS = ['small-n-mixed', 'mid-n-mixed', 'medium-n-mixed', 'big-n-mixed']


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prior family sensitivity experiment.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names (YAML files in evaluation/configs/)')
    parser.add_argument('--n_samples', type=int, default=512, help='posterior samples')
    parser.add_argument('--valid', action='store_true', help='use validation set instead of test set')
    parser.add_argument('--delta', action='store_true', help='plot delta variance instead of absolute')
    return parser.parse_args()
# fmt: on


def loadEvalConfig(name: str) -> argparse.Namespace:
    """Load an evaluation YAML config."""
    path = EVAL_CFG_DIR / f'{name}.yaml'
    assert path.exists(), f'eval config not found: {path}'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['name'] = name
    return argparse.Namespace(**cfg)


def initModel(cfg: argparse.Namespace) -> tuple[Approximator, dict]:
    """Load model and return it alongside the valid data config."""
    data_cfg = loadDataConfig(cfg.d_tag)
    assimilateConfig(cfg, data_cfg)
    data_cfg_valid = loadDataConfig(cfg.d_tag_valid)

    model_cfg_path = METABETA / 'models' / 'configs' / f'{cfg.m_tag}.yaml'
    model_cfg = modelFromYaml(
        model_cfg_path,
        d_ffx=cfg.max_d,
        d_rfx=cfg.max_q,
        likelihood_family=getattr(cfg, 'likelihood_family', 0),
    )
    model = Approximator(model_cfg)

    run = runName(vars(cfg))
    ckpt_path = CKPT_DIR / run / f'{cfg.prefix}.pt'
    assert ckpt_path.exists(), f'checkpoint not found: {ckpt_path}'
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model_state'])
    epoch = payload.get('epoch', '?')
    print(f'  loaded {cfg.prefix} checkpoint (epoch {epoch}) from {ckpt_path}')

    model.eval()
    return model, data_cfg_valid


def getFullBatch(data_cfg: dict, partition: str) -> dict[str, torch.Tensor]:
    """Load the full batch for a given partition."""
    data_fname = datasetFilename(data_cfg, partition)
    data_path = METABETA / 'outputs' / 'data' / data_fname
    if partition == 'test':
        data_path = data_path.with_suffix('.fit.npz')
    assert data_path.exists(), f'data not found: {data_path}'
    return Dataloader(data_path).fullBatch()


def resetRng(model: Approximator, seed: int) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    model.posterior_g.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore
    model.posterior_l.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore


@torch.inference_mode()
def getPosteriorStats(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run inference and return posterior mean and SD per parameter group (rescaled)."""
    resetRng(model, seed)
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
    seed: int,
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
            stats_per_family[name] = getPosteriorStats(model, b, n_samples, seed)

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
    seed: int,
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
            stats = getPosteriorStats(model, b, n_samples, seed)
            results[param_key][name] = stats[param_key]['sd']
    return results


def aggregateTailStats(
    stats_list: list[dict[str, dict[str, torch.Tensor]]],
    n_list: list[torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Flatten and concatenate tail stats across configs.

    Different configs may have different param dimensions (d, q), so tensors
    are flattened before concatenation. Returns aggregated stats and a matching
    n tensor per param key (with each dataset's n repeated once per param dim).
    """
    agg: dict[str, dict[str, torch.Tensor]] = {}
    n_per_param: dict[str, torch.Tensor] = {}
    for param_key in stats_list[0]:
        agg[param_key] = {}
        n_flat_parts = []
        for ts, n in zip(stats_list, n_list):
            any_fam = next(iter(ts[param_key].values()))
            n_dims = any_fam.shape[-1] if any_fam.dim() > 1 else 1
            n_flat_parts.append(n.repeat_interleave(n_dims))
        n_per_param[param_key] = torch.cat(n_flat_parts)
        for family_name in stats_list[0][param_key]:
            agg[param_key][family_name] = torch.cat(
                [ts[param_key][family_name].flatten() for ts in stats_list]
            )
    return agg, n_per_param


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
    'sigma_rfx': r'$\sigma_\alpha$',
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


def _pointSizes(n_flat: np.ndarray) -> np.ndarray:
    """Map observation counts to marker sizes (already repeated per param dim)."""
    n_min, n_max = n_flat.min(), n_flat.max()
    if n_max > n_min:
        return np.interp(n_flat, (n_min, n_max), SIZE_RANGE)
    return np.full_like(n_flat, np.mean(SIZE_RANGE), dtype=float)


def plotMicro(
    tail_stats: dict[str, dict[str, torch.Tensor]],
    n_per_param: dict[str, torch.Tensor],
    n_raw: torch.Tensor,
    out_path: Path | None = None,
    delta: bool = False,
) -> None:
    """Three panels: x = posterior variance (lighter family).
    y = posterior variance (heavier family) if delta=False, else delta variance.
    Colors distinguish parameter groups; point size reflects total n."""
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
            sizes = _pointSizes(n_per_param[pk].float().numpy())
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
            # if 'ffx' in param_keys:
            #     lo, hi = 0.0, 3.8
            ax.plot([lo, hi], [lo, hi], '--', color='grey', lw=1, alpha=0.6)
            step = 1.0
            ticks = np.arange(np.floor(lo / step) * step, hi + step, step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        # stats — SD-ratio matches checkTailDirection output
        sd_light_all = np.concatenate(
            [tail_stats[pk][light].flatten().numpy() for pk in param_keys]
        )
        sd_heavy_all = np.concatenate(
            [tail_stats[pk][heavy].flatten().numpy() for pk in param_keys]
        )
        sd_ratio = sd_heavy_all.mean() / (sd_light_all.mean() + 1e-12)
        larger = sd_heavy_all > sd_light_all
        above_1 = sd_light_all > 1
        frac_above = larger[above_1].mean()
        ylabel = rf'$\Delta$ SD({heavy_lbl})' if delta else f'SD({heavy_lbl})'
        niceify(
            ax,
            {
                'title': title,
                'title_fs': 22,
                'xlabel': f'SD({light_lbl})',
                'xlabel_fs': 20,
                'ylabel': ylabel,
                'ylabel_fs': 20,
                'show_legend': False,
                'stats': {'SD-ratio': sd_ratio, '% above': frac_above * 100},
                'stats_suffix': '',
                'stats_loc_x': 0.63,
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
    # size legend based on raw n (one value per dataset, not repeated)
    n_np = n_raw.float().numpy()
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
    args = setup()
    partition = 'valid' if args.valid else 'test'
    print(f'Prior Family Sensitivity Experiment')
    print(f'Configs: {args.configs} | partition: {partition}')

    all_tail_stats = []
    all_n = []

    for config_name in args.configs:
        print(f'\n{"=" * 60}')
        print(f'Config: {config_name}')
        print(f'{"=" * 60}')

        cfg = loadEvalConfig(config_name)
        model, data_cfg_valid = initModel(cfg)
        batch = getFullBatch(data_cfg_valid, partition)
        print(f'  datasets={batch["y"].shape[0]}')

        # checkPosteriorChanges(model, batch, args.n_samples, cfg.seed)
        tail_stats = collectTailStats(model, batch, args.n_samples, cfg.seed)
        all_tail_stats.append(tail_stats)
        all_n.append(batch['n'])

    agg_tail_stats, n_per_param = aggregateTailStats(all_tail_stats, all_n)
    n_raw = torch.cat(all_n, dim=0)

    checkTailDirection(agg_tail_stats)

    out_path = DIR / 'plots'
    out_path.mkdir(exist_ok=True)
    plotMicro(agg_tail_stats, n_per_param, n_raw, out_path, delta=args.delta)

    print('\nDone.')
