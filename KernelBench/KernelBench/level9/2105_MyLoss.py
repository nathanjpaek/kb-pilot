import torch
from torch import nn
import torch.utils.data


class MyLoss(nn.Module):

    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, truth):
        return torch.sum((pred - truth) ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
