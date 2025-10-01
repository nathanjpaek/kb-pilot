import torch
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class SpatialAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d((1, in_channels))
        self.avgpool = nn.AvgPool2d((1, in_channels))
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7,
            padding=3)

    def forward(self, x):
        maxpool = self.maxpool(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        avgpool = self.avgpool(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        convolved = self.conv(maxpool + avgpool)
        out = self.activation(convolved)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
