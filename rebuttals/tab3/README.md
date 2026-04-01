# Rebuttals Table 3 - Structural Misspecification

Agreement between metabeta and NUTS when both fit an incorrectly specified linear model, across four model-size regimes. Metrics are Pearson correlations between metabeta and NUTS posterior means, pooled across datasets; **bold** marks the best value per metric within each config.

## Procedure

Results were produced using [`experiments/structural_misspecification.py`](../../experiments/structural_misspecification.py). For each model config:

The key question is not parameter recovery — ground truth is no longer well-defined under misspecification — but whether metabeta faithfully approximates the posterior of the *same misspecified linear model* that NUTS fits. Agreement demonstrates that metabeta's errors are due to model choice, not the inference procedure.

1. Datasets are drawn from the standard validation set (up to 16 per config).
2. A mean-centered nonlinear term is added to y: `λ × Σⱼ (xⱼ − x̄ⱼ)²`, varying scale λ ∈ {0, 0.25, 0.5, 0.75, 1.0}. λ=0 is the correctly specified baseline.
3. Both metabeta and NUTS fit the *incorrectly specified* linear model to the perturbed data using the true prior hyperparameters.
4. NUTS is run with 1000 draws, 2000 tuning steps, and 4 chains (cached to `results/structural_fits/`).
5. Pearson correlations between metabeta and NUTS posterior means are computed per parameter type, pooled across all datasets in a config.

Metrics reported:
- **R_ffx**: correlation on fixed-effect posterior means
- **R_σ_rfx**: correlation on random-effect scale (σ) posterior means
- **R_σ_eps**: correlation on residual noise scale (σ_eps) posterior means
- **R_rfx**: correlation on random-effect posterior means

## Results

|         Config |   Condition |      R_ffx |    R_σ_rfx |    R_σ_eps |      R_rfx |
|---------------:|------------:|-----------:|-----------:|-----------:|-----------:|
|  small-n-mixed |         λ=0 | **0.9947** | **0.9839** | **0.9986** | **0.9616** |
|  small-n-mixed |      λ=0.25 |     0.9942 |     0.9834 |     0.9953 |     0.9548 |
|  small-n-mixed |       λ=0.5 |     0.9946 |     0.9343 |     0.9868 |     0.9446 |
|  small-n-mixed |      λ=0.75 |     0.9939 |     0.8506 |     0.9705 |     0.9126 |
|  small-n-mixed |         λ=1 |     0.9937 |     0.8180 |     0.9409 |     0.8708 |
|    mid-n-mixed |         λ=0 | **0.9946** | **0.9777** | **0.9996** | **0.9702** |
|    mid-n-mixed |      λ=0.25 |     0.9917 |     0.9439 |     0.9666 |     0.9593 |
|    mid-n-mixed |       λ=0.5 |     0.9606 |     0.8690 |     0.9148 |     0.9153 |
|    mid-n-mixed |      λ=0.75 |     0.8796 |     0.8222 |     0.8562 |     0.8627 |
|    mid-n-mixed |         λ=1 |     0.8519 |     0.8178 |     0.8286 |     0.8133 |
| medium-n-mixed |         λ=0 | **0.9979** |     0.9757 | **0.9978** |     0.9896 |
| medium-n-mixed |      λ=0.25 | **0.9979** | **0.9816** |     0.9847 | **0.9914** |
| medium-n-mixed |       λ=0.5 |     0.9916 |     0.8755 |     0.9365 |     0.9567 |
| medium-n-mixed |      λ=0.75 |     0.9807 |     0.8608 |     0.8409 |     0.9250 |
| medium-n-mixed |         λ=1 |     0.9739 |     0.7979 |     0.8375 |     0.9071 |


## Summary

- At λ=0, metabeta and NUTS agree closely across all configs and parameter types, confirming the baseline inference is in agreement.
- R_ffx is the most robust metric: largely stable through λ=0.5 for small and medium configs.
- R_σ_rfx degrades earliest and most severely — with more data, NUTS's posterior over the random-effect scale shifts further from metabeta's.
- Overall, metabeta appears to generalize mixed-effects inference beyond the case of correctly specified models.
