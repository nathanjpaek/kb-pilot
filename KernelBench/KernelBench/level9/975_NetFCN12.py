import torch
import torch.nn as nn
import torch.nn.functional as F


class NetFCN12(nn.Module):

    def __init__(self):
        super(NetFCN12, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv3 = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(F.relu(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
