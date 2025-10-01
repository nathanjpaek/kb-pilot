import torch
import torch.nn as nn
import torch.utils.data


class MSELoss(nn.Module):

    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.sum((input - target) ** 2, 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
