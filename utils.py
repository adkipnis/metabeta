import torch

def symmetricMatrix2Vector(matrix: torch.Tensor) -> torch.Tensor:
    r, c = matrix.shape
    indices = torch.tril_indices(row=r, col=c, offset=0)
    return matrix[indices[0], indices[1]]

def symmetricMatrixFromVector(vector: torch.Tensor, n: int) -> torch.Tensor:
    matrix = torch.zeros(n, n)
    indices = torch.tril_indices(row=n, col=n, offset=0)
    matrix[indices[0], indices[1]] = vector
    return matrix + matrix.t() - torch.diag(matrix.diag())
