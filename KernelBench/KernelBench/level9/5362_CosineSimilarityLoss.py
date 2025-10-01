import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):

    def __init__(self, dim=1, eps=1e-08):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
        self.eps = eps

    def forward(self, inputs, target):
        scores = self.cos(inputs, target)
        return 1.0 - torch.abs(scores).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
