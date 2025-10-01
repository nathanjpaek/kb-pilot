import torch
import torch.nn as nn


class FourLayerSemSegNetWideView(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, 6, kernel_size=3, padding=
            1, stride=1)
        self.conv1d100 = torch.nn.Conv2d(in_channel, 2, kernel_size=3,
            padding=101, stride=1, dilation=101)
        self.conv2d1 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=2,
            stride=1, dilation=2)
        self.conv2d5 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=6,
            stride=1, dilation=6)
        self.conv3d0 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=1
            )
        self.conv3d3 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=4,
            stride=1, dilation=4)
        self.conv4 = torch.nn.Conv2d(8, out_channel, kernel_size=3, padding
            =1, stride=1)
        self.ReLU1 = torch.nn.ReLU()
        self.ReLU2 = torch.nn.ReLU()
        self.ReLU3 = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(8, track_running_stats=False,
            momentum=1.0)
        self.batchnorm2 = torch.nn.BatchNorm2d(8, track_running_stats=False,
            momentum=1.0)
        self.batchnorm3 = torch.nn.BatchNorm2d(8, track_running_stats=False,
            momentum=1.0)

    def forward(self, x):
        x1a = self.conv1(x)
        x1b = self.conv1d100(x)
        x1 = torch.cat((x1a, x1b), dim=1)
        x1 = self.batchnorm1(x1)
        x1 = self.ReLU1(x1)
        x2a = self.conv2d1(x1)
        x2b = self.conv2d5(x1)
        x2 = torch.cat((x2a, x2b), dim=1)
        x2 = self.batchnorm2(x2)
        x2 = self.ReLU2(x2)
        x3a = self.conv3d0(x2)
        x3b = self.conv3d3(x2)
        x3 = torch.cat((x3a, x3b), dim=1)
        x3 = self.batchnorm3(x3)
        x3 = self.ReLU3(x3)
        x4 = self.conv4(x3)
        xout = self.softmax(x4)
        return xout


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
