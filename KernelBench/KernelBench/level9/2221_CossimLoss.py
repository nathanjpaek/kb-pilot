import torch
import torch.nn as nn


class CossimLoss(nn.Module):

    def __init__(self, dim: 'int'=1, eps: 'float'=1e-08):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim, eps)

    def forward(self, output, target):
        return -self.cos_sim(output, target).mean() + 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
