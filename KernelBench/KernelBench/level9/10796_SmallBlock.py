import torch
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class SmallBlock(nn.Module):

    def __init__(self, channels):
        super(SmallBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
            kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
