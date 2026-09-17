# Evidence check: IS log marginal likelihood vs bridge sampling

Working doc for the `evidence-check` branch (started 2026-09-17). Scoping prompt:
`experiments/HANDOFF_model_comparison.md`. Script: `experiments/posthoc/evidence.py`.
Results land in `experiments/results/evidence/`.

## Goal

Protect the Discussion claim that the mean marginal IS weight is an unbiased estimate of
p(D) and that Bayes factors cost one batched forward pass. Target: one appendix table
(per size regime: median |Δ log p(D)|, BF category agreement, pool size) plus one
sentence in the outlook — or a softened sentence if agreement is poor.

## Implementation (done)

- `ImportanceSampler.getImportanceWeights` stores `log_w_raw` and
  `log_evidence = logsumexp(log_w) - log S` before PSIS smoothing; `Proposal.log_evidence`.
- `preprocessing.logJacobianStandardization(data) = -n log sd_y` moves the evidence from the
  standardized space (where the flow density lives) to the raw response scale; it cancels
  in Bayes factors on the same data. NB: the evidence must be computed on *unrescaled*
  batches and proposals — `Proposal.rescale` does not adjust `log_prob_g`.
- Reference: Meng–Wong iterative bridge sampling on the cached NUTS draws in unconstrained
  coordinates (β, log σ, z_corr) with a Gaussian proposal fitted on one half of the draws
  and evaluated on the other (swap = noise floor). Target density reuses
  `ImportanceSampler.unnormalizedPosterior`. ~60 lines, no R.
- Second reference: the same bridge on MB-IMH draws (exact marginal target) — needed for
  the reduced model in the nested comparison, validated against the NUTS bridge on the
  full model.

## Finding while verifying "why it holds": LKJ prior coordinates

The flow's `log_prob_g` is a density over the *stored* constrained correlations r
(`approximator._postprocess` subtracts `logDetJacobianCorr(z, q_max)`), but the prior term
`logProbCorrRfx(z)` in `_logPriorGlobals` is a density over unconstrained z (LKJCholesky(L)
+ |dL/dz|), evaluated in q_max dimensions. Consequences for every correlated dataset:

1. Coordinate mismatch: the weight carries a spurious factor |dr/dz|. For q = 2 this is
   exactly (1 − ρ²) — a tilt of IS/IMH posteriors toward ρ = 0.
2. Padded dimension (q_i < q_max, i.e. medium and up): the LKJCholesky exponent of L_ii is
   2(η−1) + q − i, so evaluating a q_i=2 dataset in q=3 adds 0.5 log(1−ρ²) plus a different
   normalizer; and `logDetJacobianCorr(z_pad, q_max)` is not the active block's
   determinant either (another 0.5 log(1−ρ²) for q_i=2 in q_max=3).

Both are per-sample terms, so they bias the corrected posterior of ρ (not only the
evidence). Implemented behind `ImportanceSampler(corr_prior_coords='r')` (also passed
through `MetropolisSampler`): prior as a density over the stored r in the dataset's own
q_i (`logProbCorrRfx(..., q_active)`, `logDetJacobianCorr(..., q_active)`). Default stays
`'z'` (legacy) so nothing in the paper pipeline changes silently; the experiment reports
both, plus the posterior mean of ρ under raw / IS('z') / IS('r') / IMH vs NUTS.

Unit tests: `tests/utils/test_importance_evidence.py`.

## Runs

- PoC (local, CPU): `uv run python experiments/posthoc/evidence.py --sizes small --n-datasets 32`
- Cluster scale-up (Alex):
  `uv run python experiments/posthoc/evidence.py --sizes small medium --n-datasets 512 --nested 128`

## Results (2026-09-17, local CPU, 512 test datasets per regime, S=4000 unless noted)

Full outputs: `experiments/results/evidence/{small,medium}_test_n512.{md,csv,png}`,
appendix table `evidence_normal.tex`, appendix text `appendix_evidence.tex`.
Runtime ≈ 0.4–0.6 s per dataset including both bridges and the nested fit, so the full
splits ran locally (small 194 s, medium 290 s); no cluster job needed.

Reference noise floor (bridge on NUTS, halves swapped): median 0.006 (small) / 0.008
(medium) nats. Bridge on MB-IMH draws vs bridge on NUTS: 0.008 / 0.017.

| | small | medium |
|---|---|---|
| IS('r') median \|Δ log p(D)\| S=1000 → 4000 | 0.012 → 0.007 | 0.026 → 0.017 |
| IS('r') q90 \|Δ\| S=4000 | 0.030 | 0.057 |
| frac k > 0.7 | 1/512 | 14/512 |
| IS('r') k ≤ 0.7: median / q90 / max | 0.007 / 0.029 / 0.21 | 0.016 / 0.049 / 1.12 |
| IS('r') k > 0.7: median / max | 0.13 / 0.13 | 0.08 / 3.8 |
| nested (random slope vs intercept only): n | 137 | 259 |
| BF Jeffreys category agreement IS('r') vs ref | 1.000 | 0.981 |
| … two independent bridges vs each other | 0.993 | 0.981 |
| median \|Δ ln BF\| IS('r') | 0.017 | 0.032 |
| legacy IS('z'): median Δ on correlated datasets | −0.27 (n=113) | −1.00 (n=187) |
| legacy IS('z'): BF category agreement | 0.934 | 0.745 |

Correlation posterior, mean |E[ρ] − E_NUTS[ρ]| over correlated datasets:

| | raw flow | IS('z') legacy | IS('r') | IMH ('r' target) |
|---|---|---|---|---|
| small | 0.039 | 0.046 | 0.014 | 0.014 |
| medium | 0.041 | 0.053 | 0.017 | 0.016 |

Conclusions
- The outlook claim holds as stated once the weights are coordinate-consistent: the
  IS evidence is at the bridge-sampling noise floor for S ≥ 1000 on 97–99 % of datasets,
  the error scales as S^-1/2, and every miss above one nat has PSIS k > 0.7 (already
  computed → flag those datasets). Bayes factors from one batched pass reproduce the
  reference category as often as two bridge estimates agree with each other.
- The legacy 'z' weights are biased on every correlated dataset (log E[1−ρ²] up to one nat
  on medium) and make the IS-corrected correlation posterior *worse* than the raw flow.
  The 'r' weights halve the raw flow's ρ error and match NUTS as well as IMH does.
  Recommendation: flip `corr_prior_coords` default to 'r' (ImportanceSampler and
  MetropolisSampler) and regenerate the paper's MB+IMH numbers for the Normal regimes
  (Corr(RFX) metrics move; nothing else should). The 'z' option can then be removed.
- Padded dimensions matter twice for q_i < q_max (LKJ normalizer/exponent and the padded
  z→r Jacobian); both are handled in 'r' mode. The same padded-Jacobian convention lives
  in `Approximator._postprocess`; it is self-consistent with 'r' mode and need not change.

Paper hand-back (done in ~/Code/metabeta-paper, uncommitted): `tables/evidence_normal.tex`,
appendix subsection `app:ev` appended to `appendices/result_details.tex`, and the outlook
sentence in `sections/discussion.tex` now cites it. `\citep{Meng.1996}` may need a bib entry.

## Ablation rerun with the corrected prior (2026-09-17, local, 512 test datasets, s=1000)

`experiments/posthoc/ablation.py --corr-coords {z,r} --only raw is isMarginal imhMarginal coldNuts
--refresh-summaries --split test`; results in
`metabeta/outputs/results/ablation/normal_{small,medium}_raw-is-isMarginal-imhMarginal-coldNuts[_corr-r].md`.
Runtime ≈ 10 min (small) / 7 min (medium) per convention.

Corr(RFX) row (R / NRMSE / ECE), legacy 'z' → corrected 'r', cold NUTS in brackets:

| | isMarginal | imhMarginal |
|---|---|---|
| small | 0.662/0.755/−0.022 → 0.664/0.750/0.038 [0.658/0.754/0.036] | 0.666/0.752/−0.029 → 0.662/0.751/0.033 |
| medium | 0.675/0.750/0.041 → 0.679/0.738/0.097 [0.678/0.738/0.082] | 0.665/0.758/0.021 → 0.675/0.742/0.080 |

Averaged over parameter groups (paper tables), imhMarginal:
small ECE −0.005 → 0.001, EACE 0.019 → 0.019 (NUTS 0.004 / 0.020);
medium r 0.905 → 0.906, NRMSE 0.367 → 0.365, ECE −0.002 → 0.008, EACE 0.027 → 0.032 (NUTS 0.906/0.365/0.014/0.031).
IMH acceptance rises slightly (0.736 → 0.746 small, 0.470 → 0.495 medium). All other rows move by ≤ 0.002.

Reading: the legacy weights tilted ρ toward zero, which happened to *lower* the Corr(RFX) ECE
below NUTS's own (NUTS itself under-covers correlations: +0.036 / +0.082). The corrected
weights reproduce NUTS on Corr(RFX) in R, NRMSE and ECE — i.e. MB matches NUTS as the paper
claims, but the legacy numbers were matching for the wrong reason on that row.

Impact on the paper: main oracle table (2 decimals) — Normal medium MB row ECE −0.01 → 0.01
and EACE 0.02 → 0.03; small unchanged. Appendix ablation table (3 decimals) — MB⁰+IS and
MB⁰+IMH rows change in ECE/EACE for both regimes. Large/huge have more padding (q_max 4/5)
and more correlated datasets, so expect larger shifts there; Bernoulli/Poisson share the
prior code through LaplaceImportanceSampler and are affected on their correlated datasets
too. Recommendation: flip the default to 'r', regenerate the Normal and GLMM MB(+IMH) rows
(evaluate.py / ablation.py on the cluster for large, huge and the GLMM families).

## Default flipped (2026-09-17, evening)

`corr_prior_coords` now defaults to `'r'` in `ImportanceSampler` and `MetropolisSampler`, so
`evaluate.py`, `oracle_posterior.py`, `real_posterior.py`, the misspecification suites and
the public API all use the corrected weights. `posthoc.importance.WEIGHTS_VERSION = 2` is
folded into the cache keys of refined posteriors and their summaries (`-w2` / `_w2`), so caches
written under the legacy weights are ignored rather than silently reused; raw MB sample caches
and NUTS/ADVI/Laplace summaries stay valid. `ablation.py --corr-coords z` reproduces the
legacy weights (results tagged `_corr-z`).
