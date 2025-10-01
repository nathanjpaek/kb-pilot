import torch
from torch import nn


class NonSquareError(ValueError):

    def __init__(self, name, size):
        super().__init__(
            'The {} parametrization can just be applied to square matrices. Got a tensor of size {}'
            .format(name, size))


class VectorError(ValueError):

    def __init__(self, name, size):
        super().__init__(
            'Cannot instantiate {} on a tensor of less than 2 dimensions. Got a tensor of size {}'
            .format(name, size))


class Symmetric(nn.Module):

    def __init__(self, lower=True):
        """
        Vector space of symmetric matrices, parametrized in terms of the upper or lower
        triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the matrix. Default: ``True``
        """
        super().__init__()
        self.lower = lower

    @staticmethod
    def frame(X, lower):
        if lower:
            return X.tril(0) + X.tril(-1).transpose(-2, -1)
        else:
            return X.triu(0) + X.triu(1).transpose(-2, -1)

    def forward(self, X):
        if len(X.size()) < 2:
            raise VectorError(type(self).__name__, X.size())
        if X.size(-2) != X.size(-1):
            raise NonSquareError(type(self).__name__, X.size())
        return self.frame(X, self.lower)

    @staticmethod
    def in_manifold(X, eps=1e-06):
        return X.dim() >= 2 and X.size(-2) == X.size(-1) and torch.allclose(X,
            X.transpose(-2, -1), atol=eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
