import torch
from torch import nn
import torch.nn.functional as F


class DilatedModel(nn.Module):

    def __init__(self, k=16):
        super(DilatedModel, self).__init__()
        self.conv1 = nn.Conv2d(1, k, 3, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(k, k, 3, stride=1, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(k, k, 3, stride=1, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(k, k, 3, stride=1, dilation=4, padding=4)
        self.conv5 = nn.Conv2d(k, k, 3, stride=1, dilation=8, padding=8)
        self.conv6 = nn.Conv2d(k, k, 3, stride=1, dilation=16, padding=16)
        self.conv7 = nn.Conv2d(k, k, 3, stride=1, dilation=1, padding=1)
        self.conv8 = nn.Conv2d(k, 1, 1, stride=1, dilation=1, padding=0)

    def forward(self, x):
        h = x
        h = F.relu(self.conv1(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv2(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv3(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv3(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv4(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv5(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv6(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv7(h))
        assert h.shape[2:] == x.shape[2:]
        h = F.relu(self.conv8(h))
        assert h.shape == x.shape
        return h


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
