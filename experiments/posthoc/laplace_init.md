# Laplace Newton init source for discrete-GLMM MB inference

**Branch:** `glmm-analytical-warmstart`

## Question

For the discrete families (Bernoulli, Poisson), MB inference (flow + IMH/Laplace) runs the
**local posterior flow** at inference, whereas the Gaussian model skips it (analytical local
via `gaussianHybrid`). But under `mode='laplace'` the flow's local rfx are used *only* to seed
the per-group Newton mode search (`init=proposal.rfx`); the delivered rfx are freshly redrawn
from the Laplace conditional `N(b*, H⁻¹)`. Since the target is log-concave in the rfx and the
search is damped + backtracked, the converged mode is init-independent. So: can we seed Newton
from the **cheap analytical rfx estimate** (`stats['blup_est']`, PQL, already computed for the
flow context) — or even from zeros — and drop the local flow forward pass at inference?

## Change

`LaplaceImportanceSampler` / `MetropolisSampler` / `runIMH` gained a `newton_init` knob
(`laplace_glmm.NewtonInit`), default `'flow'` (no behaviour change):

- `flow`       — the flow's local rfx draws `proposal.rfx` (historical default)
- `analytical` — `stats['blup_est']` broadcast over the sample axis (reuses the context fit)
- `cold`       — zeros

(Named `newton_init`, not `warm_start`, to avoid collision with warm-started NUTS.)

## Findings (cluster, s=4000)

### Cost — timing (`laplace_init_timing.py`, Xeon CPU node, s=4000, 24 datasets/regime)

Per-dataset ms; `local A` = local posterior flow, `lap B` = Laplace mode-search + IMH + redraw,
`MB_skip` = globals-only draw + Laplace (local flow stage removed). `summ` (~20–135 ms) and
`global` (~270–550 ms) are fixed per regime and folded into the `MB_*` totals.

| family | size | regime | m | local A | lap B | MB_full | MB_skip | save |
|---|---|---|--:|--:|--:|--:|--:|--:|
| bernoulli | small  | deep |   5 |  1527 |  738 |  2703 | 1176 | **57%** |
| bernoulli | small  | wide | 192 | 52537 | 5190 | 58167 | 5631 | **90%** |
| bernoulli | medium | deep |  10 |  2001 | 1085 |  3539 | 1538 | **57%** |
| bernoulli | medium | wide | 182 | 40081 | 7680 | 48242 | 8161 | **83%** |
| bernoulli | large  | deep |  16 |  3375 | 1506 |  5341 | 1967 | **63%** |
| bernoulli | large  | wide | 188 | 40891 | 6147 | 47491 | 6600 | **86%** |
| bernoulli | huge   | deep |  20 |  6066 | 2829 |  9400 | 3334 | **65%** |
| bernoulli | huge   | wide | 190 | 53047 | 7314 | 61026 | 7979 | **87%** |
| poisson   | small  | deep |   5 |  1504 |  770 |  2712 | 1208 | **56%** |
| poisson   | small  | wide | 192 | 52160 | 4365 | 56953 | 4792 | **92%** |
| poisson   | medium | deep |  10 |  1875 |  944 |  3173 | 1298 | **59%** |
| poisson   | medium | wide | 182 | 33035 | 6173 | 39596 | 6561 | **83%** |
| poisson   | large  | deep |  16 |  3120 | 1652 |  5211 | 2092 | **60%** |
| poisson   | large  | wide | 188 | 40508 | 6964 | 47959 | 7452 | **85%** |
| poisson   | huge   | deep |  20 |  3885 | 1840 |  6200 | 2316 | **63%** |
| poisson   | huge   | wide | 190 | 49773 | 7624 | 57813 | 8040 | **86%** |

The local flow is by far the dominant stage at the production pool (s=4000) and scales ~linearly
with the group count `m`: deep (m≈5–20) spends 1.5–6 s/dataset in the flow, wide (m≈180–192) spends
**33–53 s/dataset**, while `summ`/`global` are fixed and negligible and `lap B` scales far more
gently. Skipping the flow removes **55–65% (deep) to 83–92% (wide)** of per-dataset MB inference,
uniformly across all four model sizes and both families — the same pattern the MacBook PoC showed at
s=512 (52–82%). In absolute terms it is an order-of-magnitude wall-clock cut on CPU in the wide
regime (e.g. bernoulli-huge wide 61.0 s → 8.0 s; poisson-small wide 57.0 s → 4.8 s); after the skip,
`MB_skip` is dominated by the Laplace refinement itself. The `accept f/a` columns (flow vs
analytical init) agree within noise in every cell, so the init source does not change IMH mixing at
production scale either. (On GPU the flow is cheap regardless — this is a CPU/MacBook deployment
concern.)

### Accuracy — ablation (s=4000, 512 datasets, `imhLaplace` × init)

RFX point recovery is the only metric the init source can move — FFX/Σ/corr/calibration/LOO come
from rfx redrawn from the Laplace conditional, so they are init-invariant by construction:

| family | size | flow R / NRMSE | analytical R / NRMSE | cold R / NRMSE |
|---|---|--:|--:|--:|
| bernoulli | small  | 0.731 / 0.676 | 0.731 / 0.676 | 0.731 / 0.675 |
| bernoulli | medium | 0.699 / 0.711 | 0.699 / 0.711 | 0.699 / 0.711 |
| bernoulli | large  | 0.691 / 0.726 | 0.692 / 0.726 | 0.691 / 0.727 |
| poisson   | small  | 0.825 / 0.556 | 0.825 / 0.557 | **0.721 / 0.773** |
| poisson   | medium | 0.807 / 0.587 | 0.806 / 0.588 | 0.807 / 0.587 |
| poisson   | large  | 0.837 / 0.555 | 0.846 / 0.533 | **0.570 / 1.570** |

**`analytical` is flow-identical everywhere** (RFX R within ±0.01, NRMSE within ±0.02; ECE/EACE,
LOO-NLL and predictive metrics match to the printed digits) → seeding Newton from `blup_est` is a
free swap for the flow. **`cold` is a false economy on Poisson**: it collapses RFX recovery on
poisson-large (R 0.837→0.570, NRMSE 0.555→1.570) and dents poisson-small (R 0.825→0.721) when
Newton-from-zeros fails to reach the mode in budget and the guard pins those groups; Bernoulli and
poisson-medium are untouched. The two `huge` cold runs OOM'd (three full conditions in one process;
the ablation writes its md only after all conditions finish, so no huge md survives), but the
verdict does not depend on them. **Conclusion: seed Newton from the analytical estimate
(`blup_est`), never from zeros.**

### Parity — `laplace_init_parity.py`

Across `flow` / `analytical` / `cold`, on both families: **0% guard-pins**, mode agreement to
rel-L2 ~1e-3, log-weight agreement to ~1e-4 nats, and identical IS ESS with IMH acceptance
within noise. The init source does not change the posterior at the scales tested (deep m≈16
and a correlated q=4 model).

## The two axes are measured by different tools

**Accuracy → the ablation harness** (results above). `imhLaplace` × `newton_init ∈ {flow,
analytical, cold}` with the flow draw kept fixed (isolates the init, not the speed change), added as
additive opt-in `--only` conditions (`imhLaplaceAna` / `imhLaplaceCold`, and the SNIS `isLaplace*`
analogs) so the default full runs and the canonical `{family}_{size}.md` paper tables are untouched.
The prediction that the finite-budget guard would bite in the **extreme-Poisson** regime held — that
is exactly where `cold` degrades (poisson-large/-small) while `analytical` stays flow-identical.

**Timing → `laplace_init_timing.py` (NOT the ablation).** The ablation cannot measure the timing
win, for three reasons: (1) it caches the full neural-posterior draw to disk, so warm runs never
execute the local flow; (2) even cold it draws the proposal once and shares it across conditions —
`newton_init` only changes the Laplace refinement afterwards; (3) its per-condition timer covers
only that refinement (stage B), not the flow draw (stage A). So the local-flow stage we want to
remove is invisible to it. `laplace_init_timing.py` is the timing counterpart by construction: it
draws (`model.estimate`) and skips (`estimateNoLocal`) the flow live and uncached, resolving each
stage. It takes `--family/--size`, so it runs per-regime on a cluster CPU node for
representative-hardware numbers.

Accuracy parity for `analytical` is now confirmed, so the next step is to wire the flow-skip into
`Approximator.backward` (placeholder local + Laplace redraw seeded from `blup_est`, mirroring the
Gaussian `analytical_local_posterior` path) for the production speed win — at which point
`model.estimate` itself can be timed with/without the local flow, with no cache confound. Whether MB⁰-discrete inference also drops the local flow is a
separate open empirical question; either way the local flow stays in training, where it is needed
for calibration (the global posterior depends on it), so this is an inference-only change.
