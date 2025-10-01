import torch
from torch import nn


class DenseConvBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int'=16,
        growth_channels: 'int'=16):
        super(DenseConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (
            1, 1))
        self.conv2 = nn.Conv2d(int(growth_channels * 1), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(int(growth_channels * 2), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(int(growth_channels * 3), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(int(growth_channels * 4), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(int(growth_channels * 5), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(int(growth_channels * 6), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(int(growth_channels * 7), out_channels, (3, 
            3), (1, 1), (1, 1))
        self.relu = nn.ReLU(True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out2_concat = torch.cat([out1, out2], 1)
        out3 = self.relu(self.conv3(out2_concat))
        out3_concat = torch.cat([out1, out2, out3], 1)
        out4 = self.relu(self.conv4(out3_concat))
        out4_concat = torch.cat([out1, out2, out3, out4], 1)
        out5 = self.relu(self.conv5(out4_concat))
        out5_concat = torch.cat([out1, out2, out3, out4, out5], 1)
        out6 = self.relu(self.conv6(out5_concat))
        out6_concat = torch.cat([out1, out2, out3, out4, out5, out6], 1)
        out7 = self.relu(self.conv7(out6_concat))
        out7_concat = torch.cat([out1, out2, out3, out4, out5, out6, out7], 1)
        out8 = self.relu(self.conv8(out7_concat))
        out8_concat = torch.cat([out1, out2, out3, out4, out5, out6, out7,
            out8], 1)
        return out8_concat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
