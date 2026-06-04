# PyPI Release Plan

## Current State

- `uv build` succeeds, but the wheel is not inference-only.
- The built wheel currently includes `archive/`, `benchmarks/`, `experiments/`, `tests/`,
  `scripts/`, and every `metabeta` subpackage.
- `pyproject.toml` uses broad setuptools discovery with `where = ["."]` and no package
  include/exclude rules.
- All dependencies are currently mandatory, including development, notebook, plotting,
  training, simulation, PyMC/R, and dataset-fetching dependencies.
- The user-facing pretrained API is centered on `metabeta.models.api.Api`.
- Diagnostics should stay eager so the first diagnostic sampling call does not pay import
  latency. This means `metabeta.evaluation.predictive` remains part of the API import path.
- `Approximator` imports analytical GLMM and Gaussian local refinement code at import time.
  This may be required for current checkpoint compatibility.

## Completed Debloat Work

- Moved `Proposal` and result/posterior helper functions to `metabeta.utils.results`.
- Updated callers to import result helpers from `metabeta.utils.results` directly.
- Removed result-helper re-exports from `metabeta.utils.evaluation`.
- Moved `bambiDefaultPriors` to `metabeta.utils.priors`.
- Updated `metabeta.utils.api` so inference-side prior resolution no longer imports through
  `metabeta.simulation.prior`.
- Kept diagnostics eager, but moved the predictive diagnostics import out of
  `Api.__init__` and the sampling path.
- Ran Blue across `metabeta tests`.

## Remaining Risk

Publishing `main` as-is would still ship a research repository as a PyPI package, not a
focused pretrained-model inference package. The next work is packaging metadata and wheel
contents, not more broad `api.py` refactoring.

## Next Steps

1. Add explicit package discovery rules.

   Include only `metabeta*` and exclude top-level research/dev directories:
   `tests*`, `experiments*`, `benchmarks*`, `archive*`, and `scripts*`.

2. Add source-distribution pruning.

   Add `MANIFEST.in` so sdists also exclude `tests`, `experiments`, `benchmarks`,
   `archive`, `scripts`, `demos`, and output/data artifacts.

3. Split dependencies.

   Keep base dependencies to pretrained inference/runtime needs only. Move development
   tools such as `blue`, `pytest`, notebooks, and Spyder kernels out of base dependencies.
   Move plotting dependencies into a plotting extra. Move training/simulation/PyMC/R and
   dataset-fetching dependencies into a research/dev extra.

4. Decide whether diagnostics stay in base.

   Current decision: diagnostics are eager and should stay warm for first diagnostic
   sampling. That implies `arviz`/diagnostic dependencies remain base unless we later add
   an explicit `diagnostics` extra and a warmup hook.

5. Add a package import surface.

   Add `metabeta/__init__.py` and expose `Api` there, while keeping
   `metabeta.models.api.Api` working.

6. Add release smoke tests.

   Add a wheel-content test that fails if excluded directories are present. Add a clean-env
   install smoke test for `from metabeta import Api` and
   `Api.from_pretrained(..., warmup=False)`.

7. Rebuild and inspect artifacts.

   Run `uv build`, inspect `dist/*.whl` and `dist/*.tar.gz`, and verify `METADATA`
   contains only intended base dependencies.

8. Publish to TestPyPI first.

   Install from TestPyPI in a fresh environment, run the import smoke test, checkpoint
   download, a small `sample()` call, and both demo notebooks if practical.
