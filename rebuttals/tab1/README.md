# Rebuttals Table 1 - Prior Misspecification

Robustness of the NPE posterior under three types of prior misspecification, evaluated on test set. Metrics are averaged across datasets; **bold** marks the best value per metric within each config.

## Procedure

Results were produced using [`experiments/prior_misspecification.py`](../../experiments/prior_misspecification.py). For each model config:

The model is evaluated on the test set under a baseline condition (correct prior) and all combinations of the following perturbations:
   - **Wrong variance** (τ×{0.33, 3.0}): prior scale hyperparameters scaled by a constant factor
   - **Wrong mean** (μ+{1, 2}σ): prior location for fixed effects shifted by a multiple of the prior width
   - **Wrong family** (fam): prior family indices rotated by +1 (e.g. Normal → Student-t, Half-Normal → Half-Student-t)

## Results

