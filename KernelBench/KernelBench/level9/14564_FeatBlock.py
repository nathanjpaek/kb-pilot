import torch
import torch.nn as nn


class FeatBlock(nn.Module):

    def __init__(self, planes=128, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1)
        self.conv2 = nn.Conv2d(planes, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(self.relu(x)))
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 128, 64, 64])]


def get_init_inputs():
    return [[], {}]
