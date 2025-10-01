import torch
import torch.nn as nn
import torch.utils.data


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1):
        super(ConvBlock, self).__init__()
        self.Mconv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding)
        self.MPrelu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        x = self.Mconv(x)
        x = self.MPrelu(x)
        return x


class StageBlock(nn.Module):
    """ L1/L2 StageBlock Template """

    def __init__(self, in_channels, inner_channels, innerout_channels,
        out_channels):
        super(StageBlock, self).__init__()
        self.Mconv1_0 = ConvBlock(in_channels, inner_channels)
        self.Mconv1_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv1_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv2_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv2_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv2_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv3_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv3_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv3_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv4_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv4_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv4_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv5_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv5_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv5_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv6 = ConvBlock(inner_channels * 3, innerout_channels,
            kernel_size=1, stride=1, padding=0)
        self.Mconv7 = nn.Conv2d(in_channels=innerout_channels, out_channels
            =out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1_1 = self.Mconv1_0(x)
        out2_1 = self.Mconv1_1(out1_1)
        out3_1 = self.Mconv1_2(out2_1)
        x_cat_1 = torch.cat([out1_1, out2_1, out3_1], 1)
        out1_2 = self.Mconv2_0(x_cat_1)
        out2_2 = self.Mconv2_1(out1_2)
        out3_2 = self.Mconv2_2(out2_2)
        x_cat_2 = torch.cat([out1_2, out2_2, out3_2], 1)
        out1_3 = self.Mconv3_0(x_cat_2)
        out2_3 = self.Mconv3_1(out1_3)
        out3_3 = self.Mconv3_2(out2_3)
        x_cat_3 = torch.cat([out1_3, out2_3, out3_3], 1)
        out1_4 = self.Mconv4_0(x_cat_3)
        out2_4 = self.Mconv4_1(out1_4)
        out3_4 = self.Mconv4_2(out2_4)
        x_cat_4 = torch.cat([out1_4, out2_4, out3_4], 1)
        out1_5 = self.Mconv5_0(x_cat_4)
        out2_5 = self.Mconv5_1(out1_5)
        out3_5 = self.Mconv5_2(out2_5)
        x_cat_5 = torch.cat([out1_5, out2_5, out3_5], 1)
        out_6 = self.Mconv6(x_cat_5)
        stage_output = self.Mconv7(out_6)
        return stage_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'inner_channels': 4, 'innerout_channels':
        4, 'out_channels': 4}]
