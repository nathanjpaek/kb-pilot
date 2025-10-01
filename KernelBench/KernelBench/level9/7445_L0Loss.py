import torch
from torch import nn


class L0Loss(nn.Module):
    """L0loss from
    "Noise2Noise: Learning Image Restoration without Clean Data"
    <https://arxiv.org/pdf/1803.04189>`_ paper.

    """

    def __init__(self, gamma=2, eps=1e-08):
        super(L0Loss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        loss = (torch.abs(pred - target) + self.eps).pow(self.gamma)
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
