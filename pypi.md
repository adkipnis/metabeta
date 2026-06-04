# PyPI Release Notes

## Current State

- `uv build` succeeds, but the wheel is not inference-only.
- The built wheel currently includes `archive/`, `benchmarks/`, `experiments/`, `tests/`,
  `scripts/`, and every `metabeta` subpackage.
- `pyproject.toml` uses broad setuptools discovery with `where = ["."]` and no
  package include/exclude rules.
- All dependencies are currently mandatory, including development, notebook, plotting,
  training, simulation, PyMC/R, and dataset-fetching dependencies.
- The user-facing pretrained API is centered on `metabeta.models.api.Api`.
- `Api` currently imports a broad runtime surface at import time: pandas, tabulate,
  preprocessing, model internals, dataloader helpers, and `Proposal`.
- Diagnostics should stay eager so the first diagnostic sampling call does not pay import
  latency. This means `metabeta.evaluation.predictive` remains part of the API import path.
- `Approximator` imports analytical GLMM and Gaussian local refinement code at import time.
  This may be required for current checkpoint compatibility.
- `metabeta.utils.api` imports `bambiDefaultPriors` from `metabeta.simulation.prior`, which
  couples inference helpers to simulation code.

## Main Risk

Publishing `main` as-is would ship a research repository as a PyPI package, not a focused
pretrained-model inference package. Users would install unnecessary dependencies and receive
internal experiment, benchmark, test, training, simulation, plotting, and evaluation code.

## Should `api.py` Be Refactored First?

Yes, but only in a targeted way. Do not start with a broad `api.py` rewrite.

First refactor the import boundaries that directly affect package size and installability:

1. Move `Proposal` and small posterior result helpers out of `metabeta.utils.evaluation`
   into a runtime-safe `metabeta.utils.results` module.
2. Move `bambiDefaultPriors` out of `metabeta.simulation.prior` into a runtime-safe
   `metabeta.utils.priors` module.
3. Keep diagnostics eager, but move the import out of `Api.__init__`/sampling paths.
4. Keep plotting methods lazy-imported or move them behind an optional plotting extra.
5. Leave model architecture and analytical refinement code alone until checkpoint
   compatibility is verified.

After that, tighten packaging metadata and dependency groups. This sequence keeps changes
reviewable and avoids breaking pretrained checkpoint loading while still removing the worst
dependency bloat.

## Packaging Follow-Up

- Add explicit setuptools package discovery rules, including only `metabeta*` and excluding
  `tests*`, `experiments*`, `benchmarks*`, `archive*`, and `scripts*`.
- Add `MANIFEST.in` pruning for source distributions.
- Move development tools such as `blue`, `pytest`, notebooks, and Spyder kernels out of base
  dependencies.
- Move plotting dependencies into extras. Evaluation diagnostics are currently an intentional
  eager API dependency.
- Move training/simulation/PyMC/R/dataset-fetching dependencies into a research or dev extra.
- Add a wheel-content smoke test that fails if excluded directories are present.
- Add a clean-environment smoke test for `from metabeta import Api` and
  `Api.from_pretrained(..., warmup=False)`.
