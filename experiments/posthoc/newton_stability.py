"""Diagnostic: warm-start stochasticity of the Laplace-marginal IMH weights.

posthoc/laplace_glmm.py warm-starts the per-group Newton mode search at the flow's
paired rfx draws (init=proposal.rfx). If Newton under-converges within n_newton
steps, the "deterministic" Laplace weight log p̂(y|θ_g) becomes a stochastic function
of the rfx draw that happened to travel with each global sample: harmless extra
weight noise in SNIS, but in IMH it wobbles the pseudo-target without pseudo-marginal
(GIMH) unbiasedness backing it (see LAPLACE_UPGRADES.md, point 4).

This script measures the effect directly on one (family, size) model:

  variant A  — init at the proposal's own flow rfx (production behaviour)
  variant B  — init at s-permuted flow rfx (same marginal law, decoupled from the
               paired global): the init-induced weight stochasticity is |lw_A − lw_B|
  variant Z  — zero init (cold start)

each at several n_newton values; reference = variant A at n_newton=30. Reported:

  1. |Δ log w| between A5 and B5 (init stochasticity at the production setting)
  2. |Δ log w| of A5 / Z5 vs the converged reference (under-convergence bias)
  3. A30 vs B30 vs Z30 agreement (Newton convergence sanity: should be ~0)
  4. the fraction of common-random-number IMH accept decisions that flip A5→B5
     and A5→A30 (does the noise/bias change what the chain actually does?)

Decision rule (LAPLACE_UPGRADES.md): median |Δlog w| ≲ 0.1 and flip rate < 1 % →
negligible, document; otherwise add a convergence check to laplaceRfxModes.

Run from repo root:
    uv run python experiments/posthoc/newton_stability.py --families poisson \
        --sizes large --n-datasets 32 --n-samples 512
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation import (  # noqa: E402
    IMH_BURNIN,
    IMH_N_CHAINS,
    RESULTS_DIR,
    buildModels,
    loadData,
    loadModel,
    loadOrSampleProposals,
)

from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler  # noqa: E402
from metabeta.utils.results import Proposal  # noqa: E402

# (variant, n_newton) pairs to evaluate; A30 is the converged reference
RUNS = [('A', 5), ('A', 10), ('A', 20), ('A', 30), ('B', 5), ('B', 30), ('Z', 5), ('Z', 30)]


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', nargs='+', default=['large'], choices=['small', 'medium', 'large', 'huge'])
    p.add_argument('--families', nargs='+', default=['poisson'], choices=['bernoulli', 'poisson'])
    p.add_argument('--split', choices=['valid', 'test'], default='test')
    p.add_argument('--prefix', type=str, default='best')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--n-datasets', type=int, default=32)
    p.add_argument('--n-samples', type=int, default=512)
    return p.parse_args()
# fmt: on


def makeVariant(p: Proposal, samples_l: torch.Tensor) -> Proposal:
    """Proposal sharing p's globals but with replaced local (rfx) samples."""
    return Proposal(
        {
            'global': p.data['global'],
            'local': {'samples': samples_l, 'log_prob': p.data['local']['log_prob']},
        },
        has_sigma_eps=p.has_sigma_eps,
        d_corr=p.d_corr,
    )


def acceptDecisions(log_w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Post-burnin IMH accept decisions (b, C, T_post) for a pooled weight vector.

    Mirrors MetropolisSampler._runChains: chains initialise at t=0 of their block and
    step through the pool; `u` supplies common uniforms so weight variants are compared
    on identical randomness.
    """
    b, s = log_w.shape
    C = IMH_N_CHAINS
    T = s // C
    lw = log_w[:, : C * T].reshape(b, C, T)
    cur = lw[:, :, 0].clone()
    decisions = []
    for t in range(1, T):
        log_alpha = (lw[:, :, t] - cur).clamp(max=0.0)
        accept = u[:, :, t].log() < log_alpha
        cur = torch.where(accept, lw[:, :, t], cur)
        if t >= IMH_BURNIN:
            decisions.append(accept)
    return torch.stack(decisions, dim=2)  # (b, C, T_post)


def _q(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x, q))


def runModel(cfg: dict, args: argparse.Namespace) -> str:
    lf = cfg['likelihood_family']
    torch.manual_seed(0)
    model, epoch = loadModel(cfg['ckpt'])
    model.to(args.device)

    lines = [f'# Newton warm-start stability — {cfg["label"]} (epoch={epoch})']
    split_name = 'test.fit.npz' if args.split == 'test' else 'valid.npz'
    data_path = cfg['data_dir'] / split_name
    items, _, _, _ = loadData(data_path, args.n_datasets)
    n_ds = len(items)
    lines.append(f'Datasets: {n_ds}  |  n_samples: {args.n_samples}  |  data: {data_path.name}')

    run_name = cfg['ckpt'].parent.name
    proposals, batches = loadOrSampleProposals(
        model,
        items,
        args.n_samples,
        cfg['data_dir'],
        args.split,
        run_name,
        data_path,
        cfg['ckpt'],
        args.prefix,
        args.batch_size,
        args.device,
    )

    # log-weights per (variant, n_newton), concatenated over sub-batches → (n_ds, s)
    t0 = time.perf_counter()
    lw: dict[tuple[str, int], list[torch.Tensor]] = {run: [] for run in RUNS}
    with torch.no_grad():
        for p, batch in zip(proposals, batches):
            s = p.n_samples
            perm = torch.randperm(s)
            variants = {
                'A': p,
                'B': makeVariant(p, p.samples_l[:, :, perm, :]),
                'Z': makeVariant(p, torch.zeros_like(p.samples_l)),
            }
            for variant, n_newton in RUNS:
                sampler = LaplaceImportanceSampler(batch, likelihood_family=lf, n_newton=n_newton)
                ll, lp = sampler.unnormalizedPosterior(variants[variant])
                lw[(variant, n_newton)].append(ll + lp - p.log_prob_g)
    lw_cat = {k: torch.cat(v, dim=0) for k, v in lw.items()}
    lines.append(f'Weight passes: {len(RUNS)}  |  time: {time.perf_counter() - t0:.1f}s\n')

    ref = lw_cat[('A', 30)]

    def deltaRow(label: str, a: torch.Tensor, b: torch.Tensor) -> str:
        d = (a - b).abs()
        med = d.median(dim=1).values  # per-dataset median over the pool
        mx = d.max(dim=1).values
        return (
            f'{label:<28}  med[q50/q90/max]: '
            f'{_q(med, 0.5):8.4f} /{_q(med, 0.9):8.4f} /{med.max():8.3f}   '
            f'max[q50/max]: {_q(mx, 0.5):8.3f} /{mx.max():9.3f}'
        )

    scale = lw_cat[('A', 5)].std(dim=1)
    lines.append('|Δ log w| per dataset (quantiles over datasets):')
    lines.append(deltaRow('A5 vs B5 (init noise)', lw_cat[('A', 5)], lw_cat[('B', 5)]))
    lines.append(deltaRow('A5 vs A30 (underconv. bias)', lw_cat[('A', 5)], ref))
    lines.append(deltaRow('A10 vs A30', lw_cat[('A', 10)], ref))
    lines.append(deltaRow('A20 vs A30', lw_cat[('A', 20)], ref))
    lines.append(deltaRow('Z5 vs A30 (cold-start bias)', lw_cat[('Z', 5)], ref))
    lines.append(deltaRow('A30 vs B30 (conv. sanity)', ref, lw_cat[('B', 30)]))
    lines.append(deltaRow('A30 vs Z30 (conv. sanity)', ref, lw_cat[('Z', 30)]))
    lines.append(f'{"context: std(log w) over pool":<28}  q50 over datasets: {_q(scale, 0.5):8.3f}')

    # Are the outlier disagreements benign (both variants give a deeply negative,
    # never-accepted weight) or dangerous (a spuriously high weight → absorbing
    # state in IMH / poisoned SNIS weight)? Report, for the top-Δ samples, each
    # variant's weight relative to its dataset's pool max.
    d_ab = (lw_cat[('A', 5)] - lw_cat[('B', 5)]).abs()
    flat_idx = d_ab.flatten().argsort(descending=True)[:8]
    lines.append('\ntop |Δ| samples (A5 vs B5): lw − pool max (benign iff both << 0):')
    rel_a = lw_cat[('A', 5)] - lw_cat[('A', 5)].max(dim=1, keepdim=True).values
    rel_b = lw_cat[('B', 5)] - lw_cat[('B', 5)].max(dim=1, keepdim=True).values
    for fi in flat_idx.tolist():
        ds, samp = divmod(fi, d_ab.shape[1])
        lines.append(
            f'  ds={ds:3d} s={samp:4d}  Δ={d_ab[ds, samp]:12.1f}  '
            f'rel_A={rel_a[ds, samp]:14.1f}  rel_B={rel_b[ds, samp]:14.1f}'
        )
    # a sample is DANGEROUS when the two inits disagree substantially AND at least
    # one init puts it near the pool max (candidate absorbing state / SNIS poison)
    dangerous = (d_ab > 100.0) & (torch.maximum(rel_a, rel_b) > -50.0)
    lines.append(f'dangerous samples (|Δ|>100 & near pool max): {int(dangerous.sum())}')

    # MH decision flips under common random numbers
    b_total, s = ref.shape
    T = s // IMH_N_CHAINS
    torch.manual_seed(1)
    u = torch.rand(b_total, IMH_N_CHAINS, T)
    dec_a5 = acceptDecisions(lw_cat[('A', 5)], u)
    flips_ab = (dec_a5 != acceptDecisions(lw_cat[('B', 5)], u)).float().mean()
    flips_ar = (dec_a5 != acceptDecisions(ref, u)).float().mean()
    lines.append('\nIMH accept-decision flip rate (common RNG, post-burnin):')
    lines.append(f'  A5 vs B5 (init noise) : {100 * flips_ab:.2f}%')
    lines.append(f'  A5 vs A30 (bias)      : {100 * flips_ar:.2f}%')
    return '\n'.join(lines)


def main() -> None:
    args = setup()
    for cfg in buildModels(args.families, args.sizes, args.prefix):
        report = runModel(cfg, args)
        print(f'\n{report}\n')
        out = RESULTS_DIR / f'newton_stability_{cfg["family"]}_{cfg["size"]}.md'
        out.write_text(report + '\n')
        print(f'[saved] {out}')


if __name__ == '__main__':
    main()
