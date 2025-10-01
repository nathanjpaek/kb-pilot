import torch
import torch.utils.data
from torch import nn


class BCELoss(nn.Module):

    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(target * torch.log(torch.clamp(input,
            min=1e-10)) + (1 - target) * torch.log(torch.clamp(1 - input,
            min=1e-10)), 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
