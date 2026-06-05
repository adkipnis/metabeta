# PyPI Release Plan

## Current State

- `uv build` succeeds with explicit setuptools package discovery.
- The wheel is scoped to runtime packages for pretrained-model use: `metabeta.models`,
  `metabeta.analytical`, `metabeta.configs`, `metabeta.datasets`, `metabeta.evaluation`,
  `metabeta.plotting`, `metabeta.posthoc`, and `metabeta.utils`.
- The wheel excludes `archive/`, `benchmarks/`, `demos/`, `experiments/`, `scripts/`,
  `tests/`, `metabeta.outputs`, `metabeta.simulation`, and `metabeta.training`.
- The sdist excludes research/runtime-output directories but keeps demo notebooks.
- Base dependencies intentionally include plotting support, LOO diagnostics via ArviZ,
  and SciPy. Research/data-generation dependencies stay optional under the `research` extra.
- `pytest` is still present in base dependencies, but no runtime import under `metabeta/`
  currently requires it. Decide whether to move it to the dev group before upload.
- The public pretrained API is centered on `metabeta.Api` and `metabeta.models.api.Api`.
- Diagnostics should stay eager so the first diagnostic sampling call does not pay import
  latency. This means `metabeta.evaluation.predictive` remains part of the API import path.
- `Approximator` imports analytical GLMM and Gaussian local refinement code at import time.
  This may be required for current checkpoint compatibility.
- Local release gate on 2026-06-05:
  - `uv run blue --check --diff metabeta tests` passed.
  - `uv build` passed and produced `dist/metabeta-0.4.1.tar.gz` and
    `dist/metabeta-0.4.1-py3-none-any.whl`.
  - `uv run pytest tests/packaging/test_release_artifacts.py` passed.
  - `uv run pytest` passed with 472 passed, 2 skipped, and 4 deselected.

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
- Archived `posthoc.warmnuts` for the first PyPI release. It depends on PyMC-backed
  simulation code that is deliberately excluded from the wheel and was failing import in
  the source tree.
- Marked the remaining `metabeta.posthoc` package as experimental research code.
- Moved research-only dependencies out of base metadata. `scamd` is now only in the
  `research` extra.
- Added `MANIFEST.in` pruning so sdists omit training/simulation/output/test/research
  directories while retaining demo notebooks.
- Added packaging tests for wheel/sdist content, dependency metadata, and import smoke checks.
- Added a GitHub Actions release-check workflow for Blue, pytest, `uv build`, and artifact tests.
- Added a manual TestPyPI publish workflow that uses PyPI trusted publishing once the
  TestPyPI project/environment is configured.
- Removed the `metabeta-evaluate` console script from package metadata for the first PyPI
  release; evaluation remains available as importable library code.
- Made `archive/posthoc/*.py` trackable while keeping the rest of `archive/` ignored.
- Current PyPI search results do not show an exact `metabeta` project name, but final
  availability can only be confirmed by the upload flow.
- Ran Blue across `metabeta tests`.

## Remaining Risk

The wheel is now focused on runtime packages. `metabeta.posthoc` remains included for
experimental research workflows (`svgd`, `metropolis`, `importance`, and `generative`) and
is explicitly marked as not production-ready.

The package still depends on large scientific/runtime libraries, especially PyTorch. This is
intentional for the first release, but TestPyPI should verify that resolver behavior and wheel
selection are acceptable in a fresh environment.

`pytest` appears unused at runtime and should probably move out of base dependencies unless
there is a deliberate reason to install it for end users.

## Next Steps

1. Finish posthoc packaging cleanup.

   `metabeta.posthoc.warmnuts` has been moved to `archive/posthoc/warmnuts.py`. Rebuild the
   artifacts and keep the packaging tests asserting that it is absent from both wheel and
   sdist. The remaining shipped posthoc modules should continue to import from an extracted
   wheel.

2. Revisit dependency metadata.

   Decide whether to move `pytest` from base dependencies to the dev dependency group. Review
   the large base runtime stack once more, especially PyTorch resolver behavior on Linux and
   macOS.

3. Test installation from the built wheel.

   In a fresh Python 3.12 environment, install the built `dist/metabeta-*.whl`, then
   run `from metabeta import Api`, `Api.from_pretrained("normal", warmup=False)`, a small
   `sample()` call, and the demo notebooks if practical.

4. Publish to TestPyPI.

   Upload the same built artifacts to TestPyPI, install from TestPyPI in a fresh environment,
   and repeat the import/checkpoint/sample smoke test.

5. Final metadata review before real PyPI.

   Confirm version, README rendering, license/classifier metadata, dependency metadata, and
   whether the PyPI package name is available.

6. Publish to PyPI.

   Upload only after the TestPyPI install and smoke tests pass.
