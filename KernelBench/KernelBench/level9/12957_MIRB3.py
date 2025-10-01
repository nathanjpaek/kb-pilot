from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        def wn(x):
            return torch.nn.utils.weight_norm(x)
        self.group_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 
            1, groups=self.groups))
        self.depth_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 
            3, padding=1, groups=in_channels))
        self.point_conv = wn(nn.Conv2d(self.in_channels, self.out_channels,
            1, groups=1))

    def forward(self, x):
        x = self.group_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ConvBlockD(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, ker_size=2):
        super(ConvBlockD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        def wn(x):
            return torch.nn.utils.weight_norm(x)
        self.group_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 
            1, groups=self.groups))
        self.depth_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 
            3, padding=ker_size, dilation=ker_size, groups=in_channels))
        self.point_conv = wn(nn.Conv2d(self.in_channels, self.out_channels,
            1, groups=1))

    def forward(self, x):
        x = self.group_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MIRB3(nn.Module):

    def __init__(self, args):
        super(MIRB3, self).__init__()
        self.c_out = args.n_feats // 2

        def wn(x):
            return torch.nn.utils.weight_norm(x)
        self.conv3_1 = ConvBlock(args.n_feats, self.c_out)
        self.convd_1 = ConvBlockD(args.n_feats, self.c_out, ker_size=3)
        self.conv3_2 = ConvBlock(args.n_feats, self.c_out)
        self.convd_2 = ConvBlockD(args.n_feats, self.c_out, ker_size=3)
        self.conv3_3 = ConvBlock(args.n_feats, self.c_out)
        self.convd_3 = ConvBlockD(args.n_feats, self.c_out, ker_size=3)
        self.conv_last = wn(nn.Conv2d(args.n_feats, args.n_feats, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        res = x
        c1_1 = self.lrelu(self.conv3_1(res))
        c2_1 = self.lrelu(self.convd_1(res))
        c1_2 = self.lrelu(self.conv3_2(torch.cat([c1_1, c2_1], 1)))
        c2_2 = self.lrelu(self.convd_2(torch.cat([c1_1, c2_1], 1)))
        c1_4 = self.lrelu(self.conv3_3(torch.cat([c1_2, c2_2], 1)))
        c2_4 = self.lrelu(self.convd_3(torch.cat([c1_2, c2_2], 1)))
        out = self.conv_last(torch.cat([c1_4, c2_4], 1))
        out = out + x
        return out


def get_inputs():
    return [torch.rand([4, 18, 64, 64])]


def get_init_inputs():
    return [[], {'args': _mock_config(n_feats=18)}]
