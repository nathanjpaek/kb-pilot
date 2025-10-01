import torch
from torch import nn
import torch.nn


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 40, kernel_size=
            (1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        return out


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
