import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class _netD_Q(nn.Module):
    """
    Second part of auxiliary network Q
    """

    def __init__(self, nd=10):
        super(_netD_Q, self).__init__()
        self.linear = nn.Linear(128, nd, bias=True)
        self.softmax = nn.LogSoftmax()
        self.nd = nd

    def forward(self, x):
        x = x.view(-1, 128)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, -1)
        return x.view(-1, self.nd, 1, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
