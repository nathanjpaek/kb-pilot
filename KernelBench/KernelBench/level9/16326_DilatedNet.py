import torch
import torchvision.transforms.functional as F
from torch.nn import functional as F
from torch import nn


class DilatedNet(nn.Module):

    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(self.filters[-1], self.filters[-1], 3,
            padding=2, dilation=2)
        self.conv2 = nn.Conv2d(self.filters[-1], self.filters[-1], 3,
            padding=4, dilation=4)
        self.conv3 = nn.Conv2d(self.filters[-1], self.filters[-1], 3,
            padding=8, dilation=8)
        self.conv4 = nn.Conv2d(self.filters[-1], self.filters[-1], 3,
            padding=16, dilation=16)

    def forward(self, x):
        fst = F.relu(self.conv1(x))
        snd = F.relu(self.conv2(fst))
        thrd = F.relu(self.conv3(snd))
        fourth = F.relu(self.conv4(thrd))
        return x + fst + snd + thrd + fourth


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'filters': [4, 4]}]
