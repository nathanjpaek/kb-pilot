import torch


class PCA_layer(torch.nn.Module):

    def __init__(self, n_pc=2):
        """
        Compute u^T S u as the optimization problem of PCA.
        
        Arguments:
            p: original dataset feature dimension
            n_pc: number of principal components or dimension of projected space,
                  defaulted to be 2
        """
        super().__init__()

    def forward(self, XV):
        """
        XV: X @ V, where V is the orthornormal column matrix
        """
        n = XV.shape[0]
        return 1 / n * torch.pow(torch.linalg.norm(XV, 'fro'), 2)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
