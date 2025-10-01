import torch
from torch import nn


class AttentionMatrix(nn.Module):
    """
    Attention Matrix (unnormalized)
    """

    def __init__(self, hidden_size):
        """
        Create a module for attention matrices. The input is a pair of
        matrices, the output is a matrix containing similarity scores between
        pairs of element in the matrices.

        Similarity between two vectors `a` and `b` is measured by
        $f(a, b) = W[a;b;ab] + C$, where:
            1. $W$ is a 1-by-3H matrix,
            2. $C$ is a bias,
            3. $ab$ is the element-wise product of $a$ and $b$.


        Parameters:
            :param: hidden_size (int): The size of the vectors

        Variables/sub-modules:
            projection: The linear projection $W$, $C$.

        Inputs:
            :param: mat_0 ([batch, n, hidden_size] Tensor): the first matrices
            :param: mat_1 ([batch, m, hidden_size] Tensor): the second matrices

        Returns:
            :return: similarity (batch, n, m) Tensor: the similarity matrices,
            so that similarity[:, n, m] = f(mat_0[:, n], mat_1[:, m])
        """
        super(AttentionMatrix, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Linear(3 * hidden_size, 1)
        return

    def forward(self, mat_0, mat_1):
        """
        Forward pass.
        """
        batch, n_0, _ = mat_0.size()
        _, n_1, _ = mat_1.size()
        mat_0, mat_1 = self.tile_to_match(mat_0, mat_1)
        mat_p = mat_0 * mat_1
        combined = torch.cat((mat_0, mat_1, mat_p), dim=3)
        projected = self.projection(combined.view(batch * n_0 * n_1, 3 *
            self.hidden_size))
        projected = projected.view(batch, n_0, n_1)
        return projected

    @classmethod
    def tile_to_match(cls, mat_0, mat_1):
        """
        Enables broadcasting between mat_0 and mat_1.
        Both are tiled to 4 dimensions, from 3.

        Shape:
            mat_0: [b, n, e], and
            mat_1: [b, m, e].

        Then, they get reshaped and expanded:
            mat_0: [b, n, e] -> [b, n, 1, e] -> [b, n, m, e]
            mat_1: [b, m, e] -> [b, 1, m, e] -> [b, n, m, e]
        """
        batch, n_0, size = mat_0.size()
        batch_1, n_1, size_1 = mat_1.size()
        assert batch == batch_1
        assert size_1 == size
        mat_0 = mat_0.unsqueeze(2).expand(batch, n_0, n_1, size)
        mat_1 = mat_1.unsqueeze(1).expand(batch, n_0, n_1, size)
        return mat_0, mat_1


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
