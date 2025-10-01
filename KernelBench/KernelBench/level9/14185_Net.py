import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x


def get_inputs():
    return [torch.rand([4, 2048, 64, 64])]


def get_init_inputs():
    return [[], {}]
