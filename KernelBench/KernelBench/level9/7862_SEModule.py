import torch
from torch import nn


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=
            in_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction,
            out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs)
        return inputs * outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
