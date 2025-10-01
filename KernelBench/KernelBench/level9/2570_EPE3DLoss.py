import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn


class EPE3DLoss(nn.Module):

    def __init__(self):
        super(EPE3DLoss, self).__init__()

    def forward(self, input, target):
        return torch.norm(input - target, p=2, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
