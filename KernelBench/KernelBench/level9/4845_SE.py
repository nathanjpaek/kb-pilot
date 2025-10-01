import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(input):
    return input * input.sigmoid()


class SE(nn.Module):

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels, kernel_size=1, bias=True
            )
        self.se2 = nn.Conv2d(se_channels, in_channels, kernel_size=1, bias=True
            )

    def forward(self, input):
        output = F.adaptive_avg_pool2d(input, (1, 1))
        output = swish(self.se1(output))
        output = self.se2(output).sigmoid()
        output = input * output
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'se_channels': 4}]
