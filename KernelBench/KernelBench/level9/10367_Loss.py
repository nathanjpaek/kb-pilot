import torch
import torch.utils.data
import torch
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x, y):
        z = (x - y) ** 2
        t = z[:, 1:].sum(dim=1)
        loss = z[:, 0] + y[:, 0] * t
        loss = loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
