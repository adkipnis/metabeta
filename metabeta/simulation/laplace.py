'''
# Add laplace.py Posterior Approximation

  ## Summary

  Implement a PyMC-backed Laplace approximation script for Bayesian GLMMs. It will reuse metabeta.utils.pymc.buildPymc() so the
  approximated posterior matches the NUTS/ADVI model, fit the MAP, compute the Hessian in transformed space, sample from the Gaussian
  Laplace posterior, transform samples back to natural parameters, and append laplace_* arrays to <partition>.fit.npz.

  ## Key Changes

  - Add metabeta/simulation/laplace.py with a batch-oriented LaplaceFitter.
  - CLI:
      - data args: --size, --family, --ds_type, --config, --partition, --epoch
      - sampling args: --draws default 1000, --chains default 4, --seed
      - fitting args: --maxeval default 5000, --optimizer default L-BFGS-B, --diagonal, --force

  - Default behavior: fit every dataset in the requested partition in one process and write/update <partition>.fit.npz.
  - Output keys:
      - laplace_ffx: (B, d_max, S)
      - laplace_sigma_rfx: (B, q_max, S)
      - laplace_rfx: (B, q_max, m_max, S)
      - laplace_sigma_eps: (B, 1, S) for Gaussian likelihood only
      - laplace_corr_rfx: (B, 1, S, q_max, q_max)
      - laplace_duration, laplace_failed, plus Hessian diagnostics such as jitter/repair flags

  - Failure handling: if MAP, Hessian, Cholesky, or sample extraction fails for a dataset, save correctly shaped NaN samples and
    laplace_failed=True.

  ## Implementation Details

  - Reuse buildPymc(ds, force_diagonal=cfg.diagonal) for exact model parity with NUTS/ADVI.
  - Run pm.find_MAP(include_transformed=True, method=..., maxeval=...).
  - Compute pm.find_hessian(map_point, model=model) as transformed-space precision.
  - Stabilize Hessian by symmetrizing and adding jitter; fall back to eigenvalue clipping if needed.
  - Draw S = draws * chains samples from N(map, H^{-1}) without explicitly inverting when Cholesky succeeds.
  - Use PyMC’s DictToArrayBijection and compiled value-variable evaluator to transform each sample back to:
      - fixed effects: Intercept, x1, ...
      - random-effect sigmas: 1|i_sigma, xj|i_sigma
      - random effects: 1|i, xj|i
      - sigma for Gaussian residual scale
      - _lkj_rfx_corr when correlated, identity otherwise

  - Update metabeta/simulation/fit.py so --method laplace dispatches to LaplaceFitter.
  - Extend metabeta/utils/padding.py and metabeta/utils/dataloader.py method loops from ('nuts', 'advi') to include laplace, so
    downstream loaders can consume the samples.

  - Extend evaluation model registration in metabeta/evaluation/evaluate.py so --models LAPLACE and --models all can compare it against
    MB/NUTS/ADVI.

  ## Test Plan

  - Add focused tests in tests/simulation/test_laplace.py.
  - Test failure result shapes for Gaussian and non-Gaussian likelihoods.
  - Test Hessian stabilization helper on positive-definite, near-singular, and indefinite matrices.
  - Add a small optional PyMC integration test using pytest.importorskip('pymc'): run a tiny normal GLMM with very small draws/chains/
    maxeval, assert laplace_* keys exist, shapes match repo conventions, and successful samples are finite.

  - Add/update dataloader tests to confirm laplace_* arrays are padded, permuted, and collated like nuts_*/advi_*.
  - Verification commands:
      - uv run pytest tests/simulation/test_laplace.py
      - uv run pytest tests/utils/test_dataloader.py
      - uv run blue --check --diff metabeta tests

  ## Assumptions

  - Use PyMC for v1 because it preserves posterior parity with existing NUTS/ADVI code and avoids duplicating constrained transforms,
    priors, likelihoods, and LKJ correlation logic.

  - laplace.py writes directly to <partition>.fit.npz; it does not create per-index files or require reintegration.
  - Existing non-laplace_* keys in .fit.npz are preserved.
  - If laplace_* keys already exist, the script aborts unless --force is provided.
'''
