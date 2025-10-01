import torch
import torch.nn as nn


class BranchNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=
            7, stride=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192,
            kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=256,
            kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.adaptivemaxpool = nn.AdaptiveMaxPool2d(output_size=8)

    def forward(self, x, printShapes=False):
        if printShapes:
            None
        x = self.conv1(x)
        if printShapes:
            None
        x = self.relu1(x)
        if printShapes:
            None
        x = self.maxpool1(x)
        if printShapes:
            None
        x = self.conv2(x)
        if printShapes:
            None
        x = self.relu2(x)
        if printShapes:
            None
        x = self.maxpool2(x)
        if printShapes:
            None
        x = self.conv3(x)
        if printShapes:
            None
        x = self.relu3(x)
        if printShapes:
            None
        x = self.adaptivemaxpool(x)
        if printShapes:
            None
        x = torch.flatten(x, 1)
        if printShapes:
            None
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
