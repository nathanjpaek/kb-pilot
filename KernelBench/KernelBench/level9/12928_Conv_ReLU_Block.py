import torch
import torch.nn as nn


class Conv_ReLU_Block(nn.Module):

    def __init__(self, channel_in):
        super(Conv_ReLU_Block, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=channel_in, out_channels=128,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=32,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_4 = nn.Conv2d(in_channels=96, out_channels=32,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out_0 = self.relu(self.conv_0(residual))
        out_1 = self.relu(self.conv_1(x))
        out_2 = self.relu(self.conv_2(out_1))
        cat_1 = torch.cat((out_1, out_2), 1)
        out_3 = self.relu(self.conv_3(cat_1))
        cat_2 = torch.cat((cat_1, out_3), 1)
        out_4 = self.relu(self.conv_4(cat_2))
        cat_3 = torch.cat((cat_2, out_4), 1)
        out = torch.add(out_0, cat_3)
        out = self.relu(self.conv_5(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel_in': 4}]
