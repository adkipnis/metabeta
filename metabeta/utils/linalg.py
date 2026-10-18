import torch


def choleskySolve(B: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Solve (L Lᵀ) X = B for X given the lower Cholesky factor L, like torch.cholesky_solve.

    Two triangular solves replace torch's routine: Colab's torch 2.11.0+cu130 ships no kernel for
    torch.cholesky_solve or torch.cholesky_inverse on its T4 GPUs (sm_75), while triangular solves
    run there (#26).
    """
    Y = torch.linalg.solve_triangular(L, B, upper=False)
    return torch.linalg.solve_triangular(L.mT, Y, upper=True)
