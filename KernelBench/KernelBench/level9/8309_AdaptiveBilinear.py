import torch
import torch.nn.functional as F
import torch.nn as nn


class AdaptiveBilinear(nn.Module):

    def __init__(self):
        super(AdaptiveBilinear, self).__init__()

    def forward(self, x1, x2):
        """
        :param x1: (b, l1, dim1)
        :param x2: (b, l2, dim2)
        :return:
        """
        assert x1.size(-1) == x2.size(-1)
        x_1 = F.softmax(x1 @ x1.transpose(1, 2), dim=-1)
        x_2 = F.softmax(x2 @ x2.transpose(1, 2), dim=-1)
        x_12 = x1 @ x2.transpose(1, 2)
        x_12 = x_1 @ x_12 @ x_2.transpose(1, 2)
        return x_12


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
