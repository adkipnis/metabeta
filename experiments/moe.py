"""
Pseudo Mixture of Experts experiment: evaluate whether permutation-based
ensembling improves amortized posterior estimation.

For each evaluation config:
    1. Load model from checkpoint
    2. Load test set (or validation set under --valid flag)
    3. For each k (number of extra permuted views):
        - Run pseudo-MoE inference on each dataset (B=1)
        - Evaluate metrics against ground truth
    4. Summarize absolute metrics and changes from baseline (k=0)
    5. Save table in markdown and LaTeX format

Usage (from experiments/):
    uv run python moe.py
    uv run python moe.py --configs toy-n
    uv run python moe.py --ks 0 3 7 15
    uv run python moe.py --valid
"""
