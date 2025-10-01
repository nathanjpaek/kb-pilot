import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class AUGLoss(nn.Module):

    def __init__(self):
        super(AUGLoss, self).__init__()

    def forward(self, x1, x2):
        b = x1 - x2
        b = b * b
        b = b.sum(1)
        b = torch.sqrt(b)
        return b.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
