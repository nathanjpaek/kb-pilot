import torch
from torch import nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CEloss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CEloss)
        Floss = self.alpha * (1 - pt) ** self.gamma * CEloss
        return Floss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
