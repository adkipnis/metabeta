from pathlib import Path

import torch

from metabeta.utils.linalg import choleskySolve

PACKAGE = Path(__file__).resolve().parents[2] / 'metabeta'


def _cholesky(*batch: int, q: int) -> torch.Tensor:
    a = torch.randn(*batch, q, q, dtype=torch.float64)
    return torch.linalg.cholesky(a @ a.mT + q * torch.eye(q, dtype=torch.float64))


def test_choleskySolve_matches_torch_for_matrix_and_vector_right_hand_sides():
    torch.manual_seed(0)
    L = _cholesky(2, 3, q=4)   # (b, s, q, q)
    eye = torch.eye(4, dtype=torch.float64).expand_as(L)
    rhs = torch.randn(2, 3, 4, 1, dtype=torch.float64)

    for B in (eye, rhs):
        torch.testing.assert_close(choleskySolve(B, L), torch.cholesky_solve(B, L))


def test_no_module_calls_torch_cholesky_solve_or_inverse():
    # Colab's torch 2.11.0+cu130 has no T4 (sm_75) kernel for either (#26)
    offenders = [
        p.relative_to(PACKAGE.parent)
        for p in PACKAGE.rglob('*.py')
        if 'cholesky_solve(' in p.read_text() or 'cholesky_inverse(' in p.read_text()
    ]

    assert offenders == []
