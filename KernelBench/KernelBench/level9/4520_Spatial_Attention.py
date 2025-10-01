import torch
import torch.nn as nn


class Spatial_Attention(nn.Module):

    def __init__(self, channels, length):
        super(Spatial_Attention, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels=2, out_channels=2,
            kernel_size=3, stride=2, padding=3 // 2)
        self.resize_bilinear = nn.Upsample([length, length], mode='bilinear')
        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv2d(2, 2, 1, 1, 0)

    def forward(self, x1, x2):
        avgout1 = torch.mean(x1, dim=1, keepdim=True)
        avgout2 = torch.mean(x2, dim=1, keepdim=True)
        x = torch.cat([avgout1, avgout2], dim=1)
        x = self.conv_3x3(x)
        x = self.resize_bilinear(x)
        x = self.conv_1x1(x)
        x1, x2 = x.split([1, 1], 1)
        return x1, x2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'length': 4}]
