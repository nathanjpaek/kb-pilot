import torch
from torch import nn


class DacBlock(nn.Module):

    def __init__(self, channel):
        super(DacBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=
            1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=
            3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=
            5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=
            1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.conv1x1(self.dilate2(x)))
        dilate3_out = self.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = self.relu(self.conv1x1(self.dilate3(self.dilate2(self
            .dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
