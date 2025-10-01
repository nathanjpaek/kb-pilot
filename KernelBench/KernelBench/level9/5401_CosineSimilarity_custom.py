import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity_custom(nn.Module):

    def __init__(self, dim: 'int'=1, eps: 'float'=1e-08):
        super(CosineSimilarity_custom, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
