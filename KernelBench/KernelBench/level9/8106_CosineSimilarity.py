import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):

    def __init__(self, dim=-1):
        super(CosineSimilarity, self).__init__()
        self.m = nn.CosineSimilarity(dim=dim)

    def forward(self, i, j):
        i = F.normalize(i, p=2, dim=-1)
        j = F.normalize(j, p=2, dim=-1)
        return self.m(i, j)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
