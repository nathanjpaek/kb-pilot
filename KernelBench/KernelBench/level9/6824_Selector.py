import torch
import torch.nn as nn
import torch.utils.data


class Selector(nn.Module):

    def __init__(self):
        super(Selector, self).__init__()
        self.conv1 = nn.Conv2d(2048 + 256, 256, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 16, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 1, 3)

    def forward(self, x):
        weights = self.relu1(self.conv1(x))
        weights = self.relu2(self.conv2(weights))
        weights = self.conv3(weights)
        return weights


def get_inputs():
    return [torch.rand([4, 2304, 64, 64])]


def get_init_inputs():
    return [[], {}]
