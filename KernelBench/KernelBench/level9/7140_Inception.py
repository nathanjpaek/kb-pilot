import torch
import torch.nn as nn


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, output_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU() if output_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Inception(nn.Module):

    def __init__(self, channel, batch_norm=False):
        super(Inception, self).__init__()
        if batch_norm is False:
            self.branch1x1 = nn.Conv2d(channel[0], channel[1], kernel_size=
                (1, 1), stride=1)
            self.branch3x3_1 = nn.Conv2d(channel[0], channel[2],
                kernel_size=(1, 1), stride=1)
            self.branch3x3_2 = nn.Conv2d(channel[2], channel[3],
                kernel_size=(3, 3), stride=1, padding=1)
            self.branch5x5_1 = nn.Conv2d(channel[0], channel[4],
                kernel_size=(1, 1), stride=1)
            self.branch5x5_2 = nn.Conv2d(channel[4], channel[5],
                kernel_size=(5, 5), stride=1, padding=2)
            self.branchM_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.branchM_2 = nn.Conv2d(channel[0], channel[6], kernel_size=
                (1, 1), stride=1)
        else:
            self.branch1x1 = BasicConv2d(channel[0], channel[1],
                kernel_size=(1, 1), stride=1)
            self.branch3x3_1 = BasicConv2d(channel[0], channel[2],
                kernel_size=(1, 1), stride=1)
            self.branch3x3_2 = BasicConv2d(channel[2], channel[3],
                kernel_size=(3, 3), stride=1, padding=1)
            self.branch5x5_1 = BasicConv2d(channel[0], channel[4],
                kernel_size=(1, 1), stride=1)
            self.branch5x5_2 = BasicConv2d(channel[4], channel[5],
                kernel_size=(5, 5), stride=1, padding=2)
            self.branchM_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.branchM_2 = BasicConv2d(channel[0], channel[6],
                kernel_size=(1, 1), stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        branch1x1 = self.relu(self.branch1x1(x))
        branch3x3_1 = self.relu(self.branch3x3_1(x))
        branch3x3_2 = self.relu(self.branch3x3_2(branch3x3_1))
        branch5x5_1 = self.relu(self.branch5x5_1(x))
        branch5x5_2 = self.relu(self.branch5x5_2(branch5x5_1))
        branchM_1 = self.relu(self.branchM_1(x))
        branchM_2 = self.relu(self.branchM_2(branchM_1))
        outputs = [branch1x1, branch3x3_2, branch5x5_2, branchM_2]
        return torch.cat(outputs, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': [4, 4, 4, 4, 4, 4, 4]}]
