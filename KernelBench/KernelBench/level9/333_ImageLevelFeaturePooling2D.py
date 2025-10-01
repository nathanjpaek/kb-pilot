import torch
from torch import nn
from torch.nn import functional as F


class ImageLevelFeaturePooling2D(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        x1 = torch.mean(x.view(x.size(0) * 2, x.size(1), -1), dim=2)
        x2 = x1.view(-1, self.out_channels, 1, 1)
        x3 = F.interpolate(x2, size=(x.size(-2), x.size(-1)))
        return x3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4}]
