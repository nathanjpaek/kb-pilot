import torch
import torch.nn as nn


class ConsinSimilarityLoss(nn.Module):

    def __init__(self, dim: 'int'=1, eps: 'float'=1e-08, min_zero: 'bool'=True
        ):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim, eps)
        self.min_zero = min_zero

    def forward(self, output: 'torch.Tensor', target: 'torch.Tensor'):
        cossim = self.criterion(output, target).mean()
        if self.min_zero:
            cossim = -cossim + 1
        return cossim


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
