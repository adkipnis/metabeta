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

## Local CPU findings

### Cost — `laplace_init_timing.py` (large model, s=512, 24 datasets/regime)

Per-dataset ms; `MB_skip` = globals-only draw + Laplace (local flow stage removed).

| family | regime | m | local A | laplace B | MB_full | MB_skip | save |
|---|---|--:|--:|--:|--:|--:|--:|
| bernoulli | deep | 16 | 86 | 54 | 164 | 78 | **52%** |
| bernoulli | wide | 188 | 1094 | 247 | 1386 | 246 | **82%** |
| poisson | deep | 16 | 86 | 39 | 149 | 56 | **63%** |
| poisson | wide | 188 | 1109 | 214 | 1368 | 267 | **81%** |

The local flow is the dominant stage and scales ~linearly with the number of groups `m`
(86 ms at m=16 → ~1100 ms at m=188); the Laplace scales far more gently. Skipping the flow
cuts per-dataset MB inference by ~52–63% (deep) to ~80–82% (wide) on CPU. (On GPU the flow is
cheap regardless — this is a CPU/MacBook-deployment concern.)

### Parity — `laplace_init_parity.py`

Across `flow` / `analytical` / `cold`, on both families: **0% guard-pins**, mode agreement to
rel-L2 ~1e-3, log-weight agreement to ~1e-4 nats, and identical IS ESS with IMH acceptance
within noise. The init source does not change the posterior at the scales tested (deep m≈16
and a correlated q=4 model).

## The two axes are measured by different tools

**Accuracy → the ablation harness.** Run `imhLaplace` × `newton_init ∈ {flow, analytical, cold}`
with the flow draw kept fixed (isolate the init, not the speed change), added as additive opt-in
`--only` conditions (`imhLaplaceAna` / `imhLaplaceCold`, and the SNIS `isLaplace*` analogs) so the
default full runs and the canonical `{family}_{size}.md` paper tables are untouched. This answers
the open question: **posterior-accuracy** parity (recovery, calibration/ECE, LOO-NLL) of
`analytical` / `cold` vs `flow`, especially in the **wide** and **extreme-Poisson** regimes where
the finite-budget guard can bite (wide-Poisson acceptance was the lowest/noisiest here).

**Timing → `laplace_init_timing.py` (NOT the ablation).** The ablation cannot measure the timing
win, for three reasons: (1) it caches the full neural-posterior draw to disk, so warm runs never
execute the local flow; (2) even cold it draws the proposal once and shares it across conditions —
`newton_init` only changes the Laplace refinement afterwards; (3) its per-condition timer covers
only that refinement (stage B), not the flow draw (stage A). So the local-flow stage we want to
remove is invisible to it. `laplace_init_timing.py` is the timing counterpart by construction: it
draws (`model.estimate`) and skips (`estimateNoLocal`) the flow live and uncached, resolving each
stage. It takes `--family/--size`, so it runs per-regime on a cluster CPU node for
representative-hardware numbers.

Only once accuracy parity is confirmed do we wire the flow-skip into `Approximator.backward`
(placeholder local + Laplace redraw, mirroring the Gaussian `analytical_local_posterior` path) for
the production speed win — at which point `model.estimate` itself can be timed with/without the
local flow, with no cache confound. Whether MB⁰-discrete inference also drops the local flow is a
separate open empirical question; either way the local flow stays in training, where it is needed
for calibration (the global posterior depends on it), so this is an inference-only change.
