import torch
import torch.nn as nn
import torch.nn.functional as F


class Pointwise(nn.Module):

    def __init__(self, Cin=4, K=1, Cout=10):
        super(Pointwise, self).__init__()
        self.conv1 = nn.Conv2d(Cin, Cout, kernel_size=K, bias=False,
            padding=0, stride=1)

    def forward(self, x):
        return F.relu(self.conv1(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
