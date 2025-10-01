import torch
from torch import nn


class Asym_ReLU_Block(nn.Module):

    def __init__(self):
        super(Asym_ReLU_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =(3, 1), stride=1, padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =(1, 3), stride=1, padding=(0, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)))


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
