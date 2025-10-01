import torch
import torch.nn as nn


class statm_loss(nn.Module):

    def __init__(self, eps=2):
        super(statm_loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        x = x.view(x.size(0), x.size(1), -1)
        y = y.view(y.size(0), y.size(1), -1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean - y_mean).pow(2).mean(1)
        return mean_gap.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
