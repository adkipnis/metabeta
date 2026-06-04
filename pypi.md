# PyPI Release Plan

## Current State

- `uv build` succeeds, but the wheel is not inference-only.
- The built wheel currently includes `archive/`, `benchmarks/`, `experiments/`, `tests/`,
  `scripts/`, and every `metabeta` subpackage.
- `pyproject.toml` uses broad setuptools discovery with `where = ["."]` and no package
  include/exclude rules.
- Base dependencies intentionally include plotting support, LOO diagnostics via ArviZ,
  SciPy, and pytest. Research/data-generation dependencies should stay optional.
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
- Added explicit setuptools package discovery with regular packages only.
- Added package marker files for the public package surface.
- Kept `metabeta.plotting` and predictive/evaluation diagnostics in the base package.
- Moved Gaussian local refinement to `metabeta.analytical.lmm.gaussian_local`.
- Removed conformal calibration support from `metabeta.evaluation.summary`.
- Archived `posthoc.conformal`, `posthoc.coordinate`, and `posthoc.laplace`.
- Marked the remaining `metabeta.posthoc` package as experimental research code.
- Moved research-only dependencies out of base metadata. `scamd` is now only in the
  `research` extra.
- Added `MANIFEST.in` pruning so sdists omit training/simulation/output/test/research
  directories while retaining demo notebooks.
- Ran Blue across `metabeta tests`.

## Remaining Risk

The wheel is now focused on runtime packages. `metabeta.posthoc` remains included for
experimental research workflows (`warmnuts`, `svgd`, `metropolis`, `importance`, and
`generative`) and is explicitly marked as not production-ready.

## Next Steps

1. Add release smoke tests.

   Add a wheel-content test that fails if excluded directories are present. Add a clean-env
   install smoke test for `from metabeta import Api` and
   `Api.from_pretrained(..., warmup=False)`.

2. Rebuild and inspect artifacts in CI.

   Run `uv build`, inspect `dist/*.whl` and `dist/*.tar.gz`, and verify `METADATA`
   contains only intended base dependencies and extras.

3. Decide how to handle research-adjacent modules inside included packages.

   The inference-required Gaussian local refinement now lives in `metabeta.analytical.lmm`.
   `posthoc.conformal`, `posthoc.coordinate`, and `posthoc.laplace` have been archived.
   Summary evaluation no longer accepts calibrators. The remaining posthoc modules are
   intentionally shipped as experimental research code.

4. Review the evaluation console script.

   `metabeta-evaluate` currently remains in package metadata because evaluation is in base.
   Decide whether PyPI should expose it or only expose the programmatic API.

5. Publish to TestPyPI first.

   Install from TestPyPI in a fresh environment, run the import smoke test, checkpoint
   download, a small `sample()` call, and both demo notebooks if practical.
