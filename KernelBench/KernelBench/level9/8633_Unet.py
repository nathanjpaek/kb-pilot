import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_bn_relu(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, has_relu=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=stride,
            padding=1, bias=True)
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        return x


class BlockIn(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(BlockIn, self).__init__()
        self.conv1 = conv_bn_relu(in_channel, out_channel)
        self.conv2 = conv_bn_relu(out_channel, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Projector(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Projector, self).__init__()
        self.conv1 = conv_bn_relu(in_channel, out_channel, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


class Blockdown(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Blockdown, self).__init__()
        self.conv1 = conv_bn_relu(in_channel, out_channel, stride=2)
        self.conv2 = conv_bn_relu(out_channel, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BlockUp(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(BlockUp, self).__init__()
        self.conv1 = conv_bn_relu(in_channel, out_channel)
        self.conv_adjust = conv_bn_relu(out_channel * 2, out_channel)

    def forward(self, feature_small, feature_big):
        feature_small = self.conv1(feature_small)
        f_resize = F.interpolate(feature_small, scale_factor=2, mode=
            'bilinear', align_corners=True)
        f_cat = torch.cat([f_resize, feature_big], 1)
        f_adjust = self.conv_adjust(f_cat)
        return f_adjust


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.unet_in = BlockIn(3, 32)
        self.unet_d1 = Blockdown(32, 64)
        self.unet_d2 = Blockdown(64, 128)
        self.unet_d3 = Blockdown(128, 256)
        self.unet_d4 = Blockdown(256, 512)
        self.unet_u0 = BlockUp(512, 256)
        self.unet_u1 = BlockUp(256, 128)
        self.unet_u2 = BlockUp(128, 64)
        self.unet_u3 = BlockUp(64, 32)
        self.unet_u4 = Projector(32, 3)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        f_d_64 = self.unet_in(x)
        f_d_32 = self.unet_d1(f_d_64)
        f_d_16 = self.unet_d2(f_d_32)
        f_d_8 = self.unet_d3(f_d_16)
        f_d_4 = self.unet_d4(f_d_8)
        f_u_8 = self.unet_u0(f_d_4, f_d_8)
        f_u_16 = self.unet_u1(f_u_8, f_d_16)
        f_u_32 = self.unet_u2(f_u_16, f_d_32)
        f_u_64 = self.unet_u3(f_u_32, f_d_64)
        p = self.Tanh(self.unet_u4(f_u_64))
        return p


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
