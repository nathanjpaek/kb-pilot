import torch
import torch.nn as nn
import torch.nn.init


class Coral(nn.Module):

    def __init__(self):
        super(Coral, self).__init__()

    def forward(self, a, b):
        """
        Arguments:
            a: a float tensor with shape [n, d].
            b: a float tensor with shape [m, d].
        Returns:
            a float tensor with shape [].
        """
        d = a.size(1)
        a = a - a.mean(0)
        b = b - b.mean(0)
        cs = torch.matmul(a.t(), a)
        ct = torch.matmul(b.t(), b)
        normalizer = 4 * d * d
        return ((cs - ct) ** 2).sum() / normalizer


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
