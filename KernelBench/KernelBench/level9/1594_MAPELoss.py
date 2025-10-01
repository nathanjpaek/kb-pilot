import torch
import torch.nn as nn


class MAPELoss(nn.Module):

    def __init__(self, eps=1e-08):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.mean(torch.abs(y - y_hat) / torch.abs(y + self.eps)) * 100


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
