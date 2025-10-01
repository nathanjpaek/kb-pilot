import torch
import torch.nn as nn


class ThreeLayerSemSegNetWideViewHighDim(nn.Module):
    """Each layer has more channels than the standard model"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, 12, kernel_size=3, padding
            =1, stride=1)
        self.conv1d100 = torch.nn.Conv2d(in_channel, 4, kernel_size=3,
            padding=101, stride=1, dilation=101)
        self.conv2d1 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=2,
            stride=1, dilation=2)
        self.conv2d5 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=6,
            stride=1, dilation=6)
        self.conv3 = torch.nn.Conv2d(16, out_channel, kernel_size=3,
            padding=1, stride=1)
        self.ReLU1 = torch.nn.ReLU()
        self.ReLU2 = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(16, track_running_stats=
            False, momentum=1.0)
        self.batchnorm2 = torch.nn.BatchNorm2d(16, track_running_stats=
            False, momentum=1.0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1d100(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.batchnorm1(x)
        x = self.ReLU1(x)
        x1 = self.conv2d1(x)
        x2 = self.conv2d5(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.batchnorm2(x)
        x = self.ReLU2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
