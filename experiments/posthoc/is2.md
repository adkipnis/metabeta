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

## Cost of a model comparison (CPU, M3, one dataset, K = 8)

Flow draws with the local flow skipped (`estimate(local=False)`) + one evidence pass, median
over 8 test datasets; a Bayes factor is two of these.

| regime | S=1000 flow + IS² | S=4000 flow + IS² | Laplace pass, S=4000 |
|:-------|-------------------:|-------------------:|---------------------:|
| small  | 0.07 s | 0.19–0.24 s | 0.07–0.10 s |
| medium | 0.08 s | 0.29 s      | 0.12 s |
| large  | 0.15 s | 0.54–0.56 s | 0.27–0.28 s |
| huge   | 0.15 s | 0.58–0.62 s | 0.25–0.32 s |

IS² ≈ 1.7× a Laplace pass (Newton modes shared; K streamed likelihood passes, no K axis in
memory). The 70 s/dataset of evidence.py is the validation (AGQ bridges on NUTS draws, three
IMH runs, diagnostic IS² passes), not the estimator.

Device: IS²/AGQ run on the model device like imhLaplace. A broadcast `solve_triangular`
(prior whitening over groups) returned wrong values / segfaulted on MPS → replaced by a
precomputed L⁻¹ and a broadcast matmul; AGQ nodes now live on the model device. MPS itself is
50–70× slower than CPU for the Laplace path too (small batched linear algebra), so it says
nothing about CUDA; CUDA untested locally (`pytest -k accelerator` on a GPU node).

## Real data: is the Bernoulli σ-ratio 0.96–0.97 the nAGQ = 1 bias? (colleague's suggestion)

Real Bernoulli sets have small clusters (median n/m = 12, min n_j ≈ 5, 18–46 % of datasets
with median n_j ≤ 10). σ-ratio = median over all active parameters of sd_MB / sd_NUTS, so
it is dominated by rfx entries.

1. Laplace *target* alone (NUTS draws tilted by exp(log p_Laplace − log p_AGQ), 64
   small-b-real datasets, no chain noise): log σ_rfx shifts by a median −0.04 posterior sd
   (q10 −0.12 … −0.16), width unchanged (sd ratio 1.00); no clear trend with n_j.
   Real but small, and it cannot narrow posteriors. medium-b-real (64, q ≤ 3): median
   −0.065 sd (q10 down to −0.22 for n_j ≤ 8), sd ratio 1.00; Laplace marginal error grows
   with q (median −0.30 vs −0.10 nats).
2. real_posterior.py, small-b-real, 449 converged datasets, n_samples = 1000:

   | method        | σ-ratio → 1 | rank-MAD | ΔLOO-NLL |
   |:--------------|------------:|---------:|---------:|
   | MB            | 1.02 ± 0.03 | 0.01 | 0.00 |
   | MB+imhLaplace | 0.97 ± 0.02 | 0.01 | 0.00 |
   | MB+imhPM      | 1.00 ± 0.01 | 0.00 | −0.00 |

   The pseudo-marginal chain removes the deficit. With (1), the narrowing comes mostly from
   the Laplace-Gaussian rfx redraw N(b*, H⁻¹) (skewed conditionals of small binary
   clusters), not from the nAGQ = 1 target; imhPM fixes both. Refinement ≈ 0.9 s/dataset
   locally (6.7 min for 449).

## Full ablation (cluster, 512 test datasets per regime, pool 4000; pulled to ~/Downloads/hpc-pull/ablation-is2)

| model | method | σ ECE | RFX ECE | RFX joint ECE | LOO-NLL | accept | s/ds |
|:--|:--|--:|--:|--:|--:|--:|--:|
| b-small | imhLaplace / imhPM / NUTS | .024 / .029 / .031 | −.024 / −.015 / −.014 | −.001 / .010 / .013 | .551 / .550 / .550 | .78 / .76 | 2.7 / 3.8 |
| b-large | imhLaplace / imhPM / NUTS | .002 / .007 / .002 | −.017 / −.014 / −.008 | −.006 / −.012 / .002 | .431 / .427 / .427 | .36 / .35 | 2.5 / 4.9 |
| b-huge  | imhLaplace / imhPM / NUTS | .007 / .006 / .012 | −.003 / −.008 / .009 | −.021 / −.047 / −.012 | .412 / .408 / .408 | .19 / .19 | 2.3 / 2.7 |
| p-large | imhLaplace / imhPM / NUTS | −.040 / −.041 / −.037 | −.024 / −.033 / −.023 | −.010 / −.025 / −.008 | 1.404 / 1.394 / 1.395 | .27 / .27 | 2.4 / 5.1 |
| p-huge  | imhLaplace / imhPM / NUTS | −.014 / −.025 / .007 | −.004 / −.023 / −.002 | −.004 / −.038 / −.001 | 1.402 / 1.393 / 1.397 | .15 / .14 | 2.4 / 2.9 |

imhPM matches NUTS's LOO-NLL in all 8 regimes (imhLaplace is slightly worse in all 8) and its
calibration is closest to NUTS at small/medium. At low acceptance (large/huge, 0.14–0.35) it
under-covers the rfx: a pseudo-marginal chain carries each state's rfx, so every rejection
duplicates the whole rfx vector, whereas imhLaplace redraws fresh rfx at each kept step.
Candidate fix: a conditional-IS refresh of the rfx per kept step (keep the state's draw, add
K − 1 fresh ones, select ∝ weight; leaves p(rfx | θ, y) invariant, ≈ one extra IS² pass).

## Checkpoint prefix

`--prefix latest` is the default we use (the checkpoint of the paper tables); evidence.py and
is2_tuning.py defaulted to `best` until 2026-09-26, so the small-regime PoC evidence and
tuning numbers above, and the published Normal evidence CSVs, come from best.pt. The cluster
evidence array runs on latest.pt.

## Open

- Scale-up (medium–huge, 512 datasets, q up to 5): AGQ cost grows as nodes^q (q=5: 243
  passes per bridge evaluation); the Laplace tail should grow with q and small groups.
- NUTS rfx draws (GBs) are not read, so the fidelity check covers β and σ_rfx only.
- Rerun small with α = 0.01 at cluster scale (PoC used 0.1).
