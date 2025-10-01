import torch
from torch import nn


class SumCombination(nn.Module):

    def __init__(self, dim_in, normalize=True):
        super(SumCombination, self).__init__()
        self.conv = nn.Conv1d(dim_in, 1, 1)
        self.normalize = normalize

    def forward(self, x, qlen):
        scores = self.conv(x.permute(0, 2, 1))[:, :, 0]
        if self.normalize:
            scores = scores.sum(dim=1) / qlen.type_as(scores)
        return scores


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4}]
