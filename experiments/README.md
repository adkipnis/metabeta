Experiments
===========

Experiment scripts are grouped to mirror `metabeta/` package areas:

- `analytical/` — analytical GLMM benchmarks and diagnostics.
- `datasets/` — dataset summary and inspection scripts.
- `evaluation/` — posterior quality, LOO, NPE, and real/oracle evaluation scripts.
- `plotting/` — scripts that primarily produce plots or runtime tables.
- `posthoc/` — post-hoc correction, local posterior, and warm-start diagnostics.
- `simulation/` — simulator, prior, and PyMC/Bambi equivalence checks.
- `utils/` — experiments for shared utility behavior such as MoE.

Shared path and model-loading helpers live in `metabeta/utils/experiments.py`.
Prefer those helpers over computing paths from `__file__` inside individual
scripts.
