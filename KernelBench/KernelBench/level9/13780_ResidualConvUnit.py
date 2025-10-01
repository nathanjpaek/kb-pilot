import torch
from torch import nn
import torch.nn.parallel


class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1,
            padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
