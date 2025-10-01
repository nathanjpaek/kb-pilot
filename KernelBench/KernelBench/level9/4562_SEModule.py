import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_sigmoid(x, slope=0.1666667, offset=0.5):
    return torch.clamp(slope * x + offset, 0.0, 1.0)


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4, name=''):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=
            in_channels // reduction, kernel_size=1, stride=1, padding=0,
            bias=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction,
            out_channels=in_channels, kernel_size=1, stride=1, padding=0,
            bias=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hard_sigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
