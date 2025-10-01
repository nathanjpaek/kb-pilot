import torch
import torch.nn as nn
import torch.utils.data


class dilation_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=
        'same_padding', dilation=1):
        super(dilation_layer, self).__init__()
        if padding == 'same_padding':
            padding = int((kernel_size - 1) / 2 * dilation)
        self.Dconv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding,
            dilation=dilation)
        self.Drelu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Dconv(x)
        x = self.Drelu(x)
        return x


class stage_block(nn.Module):
    """This class makes sure the paf and heatmap branch out in every stage"""

    def __init__(self, in_channels, out_channels):
        super(stage_block, self).__init__()
        self.Dconv_1 = dilation_layer(in_channels, out_channels=64)
        self.Dconv_2 = dilation_layer(in_channels=64, out_channels=64)
        self.Dconv_3 = dilation_layer(in_channels=64, out_channels=64,
            dilation=2)
        self.Dconv_4 = dilation_layer(in_channels=64, out_channels=32,
            dilation=4)
        self.Dconv_5 = dilation_layer(in_channels=32, out_channels=32,
            dilation=8)
        self.Mconv_6 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=1, padding=0)
        self.Mrelu_6 = nn.ReLU(inplace=True)
        self.paf = nn.Conv2d(in_channels=128, out_channels=14, kernel_size=
            1, padding=0)
        self.heatmap = nn.Conv2d(in_channels=128, out_channels=9,
            kernel_size=1, padding=0)

    def forward(self, x):
        x_1 = self.Dconv_1(x)
        x_2 = self.Dconv_2(x_1)
        x_3 = self.Dconv_3(x_2)
        x_4 = self.Dconv_4(x_3)
        x_5 = self.Dconv_5(x_4)
        x_cat = torch.cat([x_1, x_2, x_3, x_4, x_5], 1)
        x_out = self.Mconv_6(x_cat)
        x_out = self.Mrelu_6(x_out)
        paf = self.paf(x_out)
        heatmap = self.heatmap(x_out)
        return paf, heatmap


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
