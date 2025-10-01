import torch
from torch import nn


class TVLoss(nn.Module):
    """Implements Anisotropic Total Variation regularization"""

    def __init__(self):
        super(TVLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x):
        X = x.detach()
        XX = x
        _b, _c, h, w = X.shape
        y_tv = self.criterion(XX[:, :, 1:, :], X[:, :, :h - 1, :])
        x_tv = self.criterion(XX[:, :, :, 1:], X[:, :, :, :w - 1])
        self.loss = y_tv + x_tv
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
