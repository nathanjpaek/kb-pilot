import torch
import torch.nn as nn
from torch.nn import functional as F


class LossD(nn.Module):

    def __init__(self, gpu=None):
        super(LossD, self).__init__()
        self.gpu = gpu
        if gpu is not None:
            self

    def forward(self, r_x, r_x_hat):
        if self.gpu is not None:
            r_x = r_x
            r_x_hat = r_x_hat
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
