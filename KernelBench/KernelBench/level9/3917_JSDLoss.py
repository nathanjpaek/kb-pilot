import math
import torch
from torch import nn


class JSDLoss(nn.Module):

    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, d_x, d_y):
        return -(math.log(2.0) + 0.5 * (torch.mean(torch.log(d_x)) + torch.
            mean(torch.log(1.0 - d_y))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
