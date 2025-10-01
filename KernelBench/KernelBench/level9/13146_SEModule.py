import torch
import torch.nn as nn
import torch.nn.functional as F


def hardsigmoid(x):
    return F.relu6(x + 3.0, inplace=True) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel //
            reduction, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel // reduction,
            out_channels=channel, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs)
        return torch.mul(inputs, outputs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
