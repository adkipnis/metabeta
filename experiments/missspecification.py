"""
Misspecification study: evaluate trained models under prior misspecification.

For each evaluation config:
    1. Load model from checkpoint
    2. Load test set (or validation set under --valid flag)
    3. For each condition (baseline + 3 misspecified):
        - Perturb prior context in the batch (variance, family, or both)
        - Sample from the model
        - Evaluate metrics against the same ground truth
    4. Summarize absolute metrics and changes from baseline
    5. Save table in markdown and LaTeX format

Conditions:
    - Baseline: original priors as stored in the data
    - Wrong variance: scale prior tau hyperparameters by a factor
    - Wrong family: rotate prior family indices (+1 mod n_families)
    - Both: wrong variance + wrong family

Usage (from experiments/):
    uv run python missspecification.py
    uv run python missspecification.py --configs toy-n
    uv run python missspecification.py --scale_factor 5.0
    uv run python missspecification.py --valid
"""
