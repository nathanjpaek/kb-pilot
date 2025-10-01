import torch
from torch import nn
import torch.nn.functional as F


class FCN(nn.Module):

    def __init__(self, k=32):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(1, k, 3, stride=2, dilation=2, padding=2)
        self.conv2 = nn.Conv2d(k, k, 3, stride=2, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(k, k, 3, stride=2, dilation=2, padding=2)
        self.up1 = nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
            output_padding=1)
        self.up2 = nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
            output_padding=1)
        self.up3 = nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
            output_padding=1)
        self.up4 = nn.ConvTranspose2d(k, 1, 1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.up1(h))
        h = F.relu(self.up2(h))
        h = F.relu(self.up3(h))
        h = self.up4(h)
        assert h.shape == x.shape
        return h


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
