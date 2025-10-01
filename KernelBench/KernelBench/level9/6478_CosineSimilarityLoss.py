import torch
import torch.nn.functional
from torch import nn


class CosineSimilarityLoss(nn.Module):

    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma

    def forward(self, output, target):
        loss = 1.0 - torch.clamp(torch.nn.functional.cosine_similarity(
            output, target, dim=1, eps=0.001), -1, +1)
        if self.gamma != 1:
            loss = torch.pow(loss, self.gamma)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
