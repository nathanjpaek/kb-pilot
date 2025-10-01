import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double 3x3 conv + relu
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        return x


class UpsampleCat(nn.Module):
    """
    Unsample input and concat with contracting tensor
    """

    def __init__(self, ch):
        super(UpsampleCat, self).__init__()
        self.up_conv = nn.Conv2d(ch, ch // 2, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)

    def forward(self, up, down):
        up = self.up_conv(up)
        up = self.up(up)
        up_w, up_h = up.size()[2:4]
        down_w, down_h = down.size()[2:4]
        dw = down_w + 4 - up_w
        dh = down_h + 4 - up_h
        down = F.pad(down, (2, 2, 2, 2))
        up = F.pad(up, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        y = torch.cat([down, up], dim=1)
        return y


class UNet(nn.Module):
    """
    UNet model
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv_1 = DoubleConv(in_channels, 64)
        self.conv_2 = DoubleConv(64, 128)
        self.conv_3 = DoubleConv(128, 256)
        self.conv_4 = DoubleConv(256, 512)
        self.conv_5 = DoubleConv(512, 1024)
        self.down = nn.MaxPool2d(2)
        self.up_1 = UpsampleCat(1024)
        self.up_2 = UpsampleCat(512)
        self.up_3 = UpsampleCat(256)
        self.up_4 = UpsampleCat(128)
        self.conv_6 = DoubleConv(1024, 512)
        self.conv_7 = DoubleConv(512, 256)
        self.conv_8 = DoubleConv(256, 128)
        self.conv_9 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(self.down(x1))
        x3 = self.conv_3(self.down(x2))
        x4 = self.conv_4(self.down(x3))
        x = self.conv_5(self.down(x4))
        x = self.conv_6(self.up_1(x, x4))
        x = self.conv_7(self.up_2(x, x3))
        x = self.conv_8(self.up_3(x, x2))
        x = self.conv_9(self.up_4(x, x1))
        x = F.pad(x, (2, 2, 2, 2))
        x = self.out_conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 256, 256])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
