import pytest
import torch

from metabeta.utils.glm import _safeSolve


def test_safe_solve_matches_torch_solve_for_well_conditioned_batch():
    A = torch.tensor(
        [
            [[3.0, 0.5], [0.5, 2.0]],
            [[2.0, -0.2], [-0.2, 1.0]],
        ]
    )
    b = torch.tensor([[1.0, 2.0], [0.5, -1.0]])

    assert torch.allclose(_safeSolve(A, b), torch.linalg.solve(A, b))


def test_safe_solve_handles_singular_batch_without_eigendecomposition(monkeypatch):
    A = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    b = torch.tensor([[2.0, 4.0], [1.0, 1.0]])

    def fail_eigvalsh(*args, **kwargs):
        raise AssertionError('_safeSolve fallback should not call eigvalsh')

    monkeypatch.setattr(torch.linalg, 'eigvalsh', fail_eigvalsh)

    solution = _safeSolve(A, b)

    assert torch.isfinite(solution).all()
    assert torch.allclose(solution[0], torch.tensor([1.0, 2.0]))


def test_safe_solve_returns_finite_values_for_nonfinite_degenerate_batch():
    A = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[float('nan'), 0.0], [0.0, 0.0]],
        ]
    )
    b = torch.tensor([[3.0, -1.0], [float('inf'), 1.0]])

    solution = _safeSolve(A, b)

    assert torch.isfinite(solution).all()
    assert torch.allclose(solution[0], torch.tensor([3.0, -1.0]))
