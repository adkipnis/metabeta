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

## Recommendation / next step (cluster)

The local experiments settle **timing** and **mechanism parity**. The open question is
**posterior-accuracy** parity (recovery, calibration/ECE, LOO-NLL, SBC) of `analytical` / `cold`
vs `flow`, especially in the **wide** and **extreme-Poisson** regimes where the finite-budget
guard can bite (wide-Poisson acceptance was the lowest/noisiest here). That is an accuracy
question the timing scripts cannot answer, so run it on the **ablation harness**
(`experiments/posthoc/ablation.py`) as an `imhLaplace` × `newton_init ∈ {flow, analytical, cold}`
comparison with the flow draw kept fixed (isolate the init, not the speed change). Wiring
`newton_init` into the harness/`posterior_eval` as additive side-by-side conditions is the
prerequisite (kept additive so it can't disturb the existing method set / paper tables).

Only once accuracy parity is confirmed do we enable the flow-skip for the speed win (and decide
separately whether MB⁰-discrete inference also drops the local flow — an open empirical
question; the local flow stays in training regardless, it is needed there for calibration).
