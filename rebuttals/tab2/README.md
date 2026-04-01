# Rebuttals Table 2 - Post-hoc Ablation

Ablation of post-hoc inference improvements over the base NPE posterior across three model-size regimes (small, mid, medium). Each condition is applied after amortized inference and evaluated on held-out test datasets. Metrics are averaged across datasets; **bold** marks the best value per metric within each config.

## Procedure

Results were produced using [`experiments/ablation.py`](../../experiments/ablation.py). For each model config:

1. Three conformal calibrators are fitted on the calibration set — one for each proposal distribution (raw, IS-corrected, SIR-resampled) — and cached to disk for reuse.
3. The test set is evaluated under six conditions:
   - **Baseline**: raw NPE posterior
   - **+ CP**: conformal prediction
   - **+ IS**: importance sampling to correct for proposal–target mismatch.
   - **+ SIR**: sampling importance resampling (rejection sampling proportional to importance weights)
   - **+ IS + CP**: the CP is based on the IS-corrected posterior
   - **+ SIR + CP**: the CP is base on the SIR-corrected posterior
4. For each condition, five metrics are computed: Pearson R, NRMSE, ECE (= observed − nominal coverage; 0 is ideal), posterior predictive NLL, and R².

## Results

|         Config |   Condition |          R |      NRMSE |         ECE |      ppNLL |         R² |
|---------------:|------------:|-----------:|-----------:|------------:|-----------:|-----------:|
|  small-n-mixed |    Baseline |     0.9682 |     0.2172 |      0.0236 |     1.3080 |     0.8581 |
|  small-n-mixed |        + CP |     0.9682 |     0.2172 |     -0.0005 |     1.3080 |     0.8581 |
|  small-n-mixed |        + IS |     0.9692 | **0.2082** | **-0.0004** |     1.2922 | **0.8683** |
|  small-n-mixed |       + SIR | **0.9696** |     0.2083 |     -0.0482 | **1.2877** | **0.8683** |
|  small-n-mixed |   + IS + CP |     0.9692 | **0.2082** |      0.2573 |     1.2922 | **0.8683** |
|  small-n-mixed |  + SIR + CP | **0.9696** |     0.2083 |      0.2566 | **1.2877** | **0.8683** |
|    mid-n-mixed |    Baseline |     0.9570 |     0.2605 |      0.0724 |     1.2120 |     0.9500 |
|    mid-n-mixed |        + CP |     0.9570 |     0.2605 |      0.0682 |     1.2120 |     0.9500 |
|    mid-n-mixed |        + IS | **0.9616** | **0.2445** |      0.0301 | **1.1161** | **0.9625** |
|    mid-n-mixed |       + SIR |     0.9602 |     0.2488 | **-0.0175** |     1.1202 |     0.9618 |
|    mid-n-mixed |   + IS + CP | **0.9616** | **0.2445** |      0.2196 | **1.1161** | **0.9625** |
|    mid-n-mixed |  + SIR + CP |     0.9602 |     0.2488 |      0.2097 |     1.1202 |     0.9618 |
| medium-n-mixed |    Baseline |     0.9442 |     0.2956 |      0.0979 |     1.3832 |     0.9355 |
| medium-n-mixed |        + CP |     0.9442 |     0.2956 |      0.0830 |     1.3832 |     0.9355 |
| medium-n-mixed |        + IS | **0.9546** |     0.2491 |      0.0240 | **1.2133** | **0.9542** |
| medium-n-mixed |       + SIR |     0.9544 | **0.2483** | **-0.0161** |     1.2407 |     0.9540 |
| medium-n-mixed |   + IS + CP | **0.9546** |     0.2491 |      0.2070 | **1.2133** | **0.9542** |
| medium-n-mixed |  + SIR + CP |     0.9544 | **0.2483** |      0.2161 |     1.2407 |     0.9540 |


## Summary

- IS and SIR consistently improve point estimates over the baseline across all configs (e.g., ppNLL drops ~0.1–0.17), with IS favoring ppNLL and SIR favoring NRMSE and ECE; neither dominates cleanly.
- CP alone provides negligible benefit, as the baseline posteriors are already near-calibrated (ECE ≈ 0–0.10).
- Combining IS/SIR with CP over-inflates intervals (ECE ~0.21–0.26) because calibrator corrections computed on the validation set do not transfer reliably under the stochasticity of IS/SIR; we will turn off CP calibration by default and keep IS instead.
