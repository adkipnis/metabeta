# IS² evidence for GLMMs (branch `is2-evidence`, working doc)

Goal: an unbiased marginal-likelihood estimate for Bernoulli/Poisson models, cheap on CPU,
validated against a NUTS-based reference as the Normal evidence was. Side question: does the
same estimator make the IMH pseudo-marginal (exact target), fixing the Laplace-IMH bias?

## Method

- `logMarginalLikelihoodIS2` (posthoc/laplace_glmm.py): importance sampling squared (IS²), Tran,
  Scharth, Pitt & Kohn, arXiv:1309.3339. Per group, K draws from the defensive mixture
  (1 − α) N(b*, H⁻¹) + α N(0, Σ) around the existing Laplace modes; p̂_j is unbiased for
  p(y_j | θ), the product over groups too, hence exp(log-evidence) is unbiased.
- The returned rfx are one weight-selected inner draw per group (reservoir sampling), which
  makes `MetropolisSampler(mode='laplace', n_inner=K)` a pseudo-marginal chain
  (Andrieu & Roberts 2009) on (θ, rfx).
- `logMarginalLikelihoodAGQ`: the same inner-weight sum with tensor-product Gauss-Hermite
  nodes (adaptive Gauss-Hermite quadrature, AGQ; lme4's nAGQ). Deterministic, converges
  geometrically; the GLMM evidence reference.

## Unit tests (tests/utils/test_is2.py)

- p̂ unbiased vs dense grid (Normal/Bernoulli/Poisson, padded group + padded rfx dim, an
  all-success Bernoulli group); SE 0.002 on the ratio, Laplace is off by 3 % on the same case.
- p̂-weighted inner draws recover E[b_j | θ, y_j] (fails when the selection is not ∝ weight).
- Tiny random-intercept Bernoulli (4 groups × 5 obs, one all-success): IS² log-evidence
  within ±0.007 nats of 2D quadrature under two proposals (Laplace: −0.047 ± 0.007).
  Posterior mean of the all-success group's intercept: truth 0.975, IS² 0.97–0.98,
  Laplace redraw 0.83.
- Pseudo-marginal IMH matches quadrature posterior means of β₀, σ, b₀ (Laplace-IMH fails
  on b₀ by 0.12).
- AGQ: Normal exact to 1e-6; Bernoulli/Poisson q=2 error 4e-2 (1 node = Laplace) →
  5e-5 (5) → 2e-7 (12 nodes).

Aside: `ImportanceSampler`'s non-PSIS branch clips log-weights at their 99 % quantile, which
biases weighted moments (bites harder under IS² noise); the evidence uses the raw weights.

## PoC ablation (small, 32 test datasets, pool 1000, `ablation.py --only raw imhLaplace imhPM coldNuts`)

Calibration against ground truth cannot separate imhPM from imhLaplace at this scale;
both sit within noise of NUTS (e.g. Bernoulli RFX ECE −0.031 PM / −0.036 Laplace / −0.030
NUTS; Poisson σ_rfx ECE 0.048 / 0.048 / 0.033). imhPM costs ≈ 1.3× imhLaplace
(0.3 vs 0.2 s/dataset at K=8) and accepts slightly less (0.71 vs 0.76 Bernoulli).
→ judge the side benefit by distance to the NUTS posterior instead (evidence.py fidelity).

## Evidence check (evidence.py --family bernoulli/poisson, small, 32 datasets)

Reference: Meng-Wong bridge on NUTS global draws with an AGQ-marginal target
(nodes/dim: q=1 20, q=2 12, q=3 5, q=4 4, q≥5 3); swapped halves as noise floor.

Run with K=8, α=0.1 (before the α default moved to 0.01), S ∈ {1000, 2000, 4000}.

| nats, S=4000                     | Bernoulli | Poisson |
|:---------------------------------|----------:|--------:|
| bridgeNuts swap noise, median    | 0.013     | 0.008   |
| IS² − bridge, median / q90 abs   | 0.009 / 0.039 | 0.009 / 0.026 |
| IS² − bridge, worst              | 0.057     | 0.079   |
| Laplace − bridge, median / q90   | 0.040 / 0.374 | 0.014 / 0.086 |
| Laplace − bridge, worst          | 0.637     | 0.114   |
| sd(log p̂) under posterior, median / q90 | 0.20 / 0.62 | 0.22 / 0.55 |
| nested ln BF: IS² − ref, median abs (n=10) | 0.039 | 0.021 |
| nested Jeffreys agreement        | 1.00      | 1.00    |

IS² sits at the reference's own noise floor, like the Normal check (small: 0.008), with no
bias (median Δ −0.003 … +0.004); Laplace has a heavy tail (Bernoulli: 3/32 datasets off by
0.4–0.6 nats). bridgeImh (pseudo-marginal IMH draws) agrees with bridgeNuts to 0.010 median,
so it is a valid reduced-model reference. ~70 s/dataset for the whole script (dominated by
the AGQ bridges and the four IMH/IS passes), not by the estimator.

Fidelity vs NUTS (median over datasets of mean-over-parameters; z = |Δ mean| / sd_NUTS):

|            | Bern. z β | Bern. z σ | Pois. z β | Pois. z σ |
|:-----------|----------:|----------:|----------:|----------:|
| raw flow   | 0.080 | 0.107 | 0.042 | 0.080 |
| imhLaplace | 0.025 | 0.053 | 0.028 | 0.028 |
| imhPM      | 0.031 | 0.033 | 0.028 | 0.028 |

Side benefit, partly: on Bernoulli the pseudo-marginal chain removes most of the Laplace
σ_rfx shift (0.053 → 0.033 sd; closer on 62 % of datasets, the floor is ≈ 1/√ESS ≈ 0.03);
on Poisson there is nothing to remove (Laplace bias below the Monte Carlo floor at ~23
obs/group). Acceptance drops 0.77 → 0.72 (Bernoulli) / 0.84 → 0.77 (Poisson) from the
weight noise. The σ/β fidelity check cannot see rfx; the unit test shows the rfx benefit
is largest for tiny all-success groups.

## IS² tuning (is2_tuning.py, small, 16 datasets, S = 1000)

median sd(log p̂) (Bernoulli; Poisson within ±0.02):

| K  | α=0   | 0.01  | 0.05  | 0.1   | s/pass |
|---:|------:|------:|------:|------:|-------:|
| 2  | 0.179 | 0.185 | 0.277 | 0.364 | 0.05 |
| 8  | 0.083 | 0.086 | 0.109 | 0.154 | 0.09 |
| 32 | 0.042 | 0.045 | 0.053 | 0.074 | 0.22 |

sd ∝ K^(−1/2); α = 0.01 costs nothing over α = 0 and keeps the bounded-weight guarantee,
α = 0.1 inflates the noise ≈ 1.8×, so the default is now α = 0.01. K = 8 gives q90
sd ≈ 0.3–0.4, i.e. log-evidence bias ≈ sd²/2 ≲ 0.001 nats after averaging over S draws.

## Open

- Scale-up (medium–huge, 512 datasets, q up to 5): AGQ cost grows as nodes^q (q=5: 243
  passes per bridge evaluation); the Laplace tail should grow with q and small groups.
- NUTS rfx draws (GBs) are not read, so the fidelity check covers β and σ_rfx only.
- Rerun small with α = 0.01 at cluster scale (PoC used 0.1).
