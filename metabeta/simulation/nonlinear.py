"""
Structural nonlinearity perturbations for the simulation prior.

Each function computes a nonlinear term to be added to the linear predictor η,
using the design matrix X. All variants operate on mean-centered non-intercept
predictors to avoid systematically shifting the intercept in expectation.

Four qualitatively distinct forms are provided:

  polynomial  — mean of squared centered predictors: captures curvature / omitted
                polynomial terms; always non-negative; symmetric; no random state.

  interaction — product of a randomly chosen pair of centered predictors: captures
                omitted multiplicative effects; sign-sensitive; zero when fewer than
                two predictors are present.

  smooth      — tanh of a random unit-norm projection of centered predictors:
                bounded saturating nonlinearity (output in (−scale, scale)); smooth
                everywhere; acts on a single linear combination.

  step        — ReLU of a random unit-norm projection of centered predictors:
                piecewise-linear hinge; always non-negative; acts on a single
                linear combination.
"""

import numpy as np

NONLINEAR_KINDS: tuple[str, ...] = ('polynomial', 'interaction', 'smooth', 'step')


def _centerPredictors(X: np.ndarray) -> np.ndarray:
    """Return non-intercept columns of X, mean-centered across observations."""
    cols = X[:, 1:]  # (n, d-1)
    return cols - cols.mean(axis=0, keepdims=True)


def _polynomial(rng: np.random.Generator, X: np.ndarray, scale: float) -> np.ndarray:
    """Mean of squared centered predictors.

    Expected magnitude ≈ 1 for standardized X (each column has variance ≈ 1),
    so `scale` is approximately the amplitude relative to σ_y.
    Always non-negative; does not consume any random state.
    """
    xc = _centerPredictors(X)
    if xc.shape[1] == 0:
        return np.zeros(X.shape[0])
    return scale * (xc**2).mean(axis=1)


def _interaction(rng: np.random.Generator, X: np.ndarray, scale: float) -> np.ndarray:
    """Product of a randomly chosen pair of centered predictors.

    For standardized uncorrelated X the product has zero mean and unit variance,
    so `scale` is approximately the amplitude relative to σ_y.
    Sign-sensitive; returns zeros when fewer than two predictors are present.
    """
    xc = _centerPredictors(X)
    d = xc.shape[1]
    if d < 2:
        return np.zeros(X.shape[0])
    j, k = rng.choice(d, size=2, replace=False)
    return scale * xc[:, j] * xc[:, k]


def _smooth(rng: np.random.Generator, X: np.ndarray, scale: float) -> np.ndarray:
    """Tanh of a random unit-norm projection of centered predictors.

    Output is bounded: term ∈ (−scale, scale). For a unit-norm projection of
    standardized predictors the argument to tanh has variance ≈ 1, giving
    typical magnitudes below scale. Returns zeros when no predictors are present.
    """
    xc = _centerPredictors(X)
    d = xc.shape[1]
    if d == 0:
        return np.zeros(X.shape[0])
    w = rng.normal(size=d)
    w /= np.linalg.norm(w) + 1e-12
    return scale * np.tanh(xc @ w)


def _step(rng: np.random.Generator, X: np.ndarray, scale: float) -> np.ndarray:
    """ReLU of a random unit-norm projection of centered predictors.

    Output is always non-negative: term ∈ [0, ∞). For a unit-norm projection of
    standardized predictors the expected value is scale / √(2π) ≈ 0.4 · scale.
    Returns zeros when no predictors are present.
    """
    xc = _centerPredictors(X)
    d = xc.shape[1]
    if d == 0:
        return np.zeros(X.shape[0])
    w = rng.normal(size=d)
    w /= np.linalg.norm(w) + 1e-12
    return scale * np.maximum(xc @ w, 0.0)


_DISPATCH: dict[str, object] = {
    'polynomial': _polynomial,
    'interaction': _interaction,
    'smooth': _smooth,
    'step': _step,
}


def adjustParamsForNonlinearity(
    params: dict[str, np.ndarray],
    X: np.ndarray,
    f_X: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Decompose f_X into a linear component absorbed by the model and an orthogonal residual.

    When a nonlinear term f(X) is added to the DGP, the misspecified linear model
    partially absorbs it: the OLS projection X @ Δβ adjusts the fixed effects, and
    only the component orthogonal to span(X) inflates the residual noise variance.

    Concretely, f_X = X @ Δβ + f_resid where Δβ = lstsq(X, f_X). The adjustments are:
      - ffx ← ffx + Δβ
      - σ_eps ← sqrt(σ_eps² + Var(f_resid))  (only when sigma_eps is present)

    Passing f_resid (not f_X) as the nonlinear_term to simulate() then leaves y
    algebraically unchanged: X @ (ffx + Δβ) + f_resid = X @ ffx + f_X. The stored
    parameters now match what an ideal Bayesian fit of the linear model — i.e. NUTS —
    would recover from the perturbed data.

    Args:
        params:  Parameter dict from Prior.sample() (keys: ffx, rfx, sigma_eps, ...).
                 Not mutated; a shallow copy with updated values is returned.
        X:       Design matrix (n, d) with intercept in column 0.
        f_X:     Nonlinear term (n,) as returned by addNonlinearity().

    Returns:
        params_adj: Adjusted copy of params with corrected ffx and sigma_eps.
        f_resid:    Residual of f_X after projecting out the linear component (n,).
                    Use this as nonlinear_term in simulate() to keep y unchanged.
    """
    delta_ffx, _, _, _ = np.linalg.lstsq(X, f_X, rcond=None)
    f_resid = f_X - X @ delta_ffx

    params_adj = {**params}
    params_adj['ffx'] = params['ffx'] + delta_ffx

    if 'sigma_eps' in params:
        sigma_eps = float(params['sigma_eps'])
        params_adj['sigma_eps'] = np.array(np.sqrt(sigma_eps**2 + float(f_resid.var())))

    return params_adj, f_resid


def addNonlinearity(
    rng: np.random.Generator,
    X: np.ndarray,
    kind: str,
    scale: float,
) -> np.ndarray:
    """Compute a nonlinear perturbation term to be added to the linear predictor η.

    Args:
        rng:   NumPy random generator (consumed by interaction, smooth, and step).
        X:     Design matrix of shape (n, d) with the intercept in column 0.
        kind:  Perturbation type — one of NONLINEAR_KINDS.
        scale: Amplitude; output magnitude is proportional to scale.

    Returns:
        term: Array of shape (n,) to add to η before sampling y.

    Raises:
        ValueError: If kind is not one of NONLINEAR_KINDS.
    """
    if kind not in _DISPATCH:
        raise ValueError(f'Unknown nonlinear kind {kind!r}; choose from {NONLINEAR_KINDS}')
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f'X must be 2-D with at least one column, got shape {X.shape}')
    return _DISPATCH[kind](rng, X, float(scale))  # type: ignore[operator]
