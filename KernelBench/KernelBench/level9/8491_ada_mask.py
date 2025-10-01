import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, ker_size, stri, pad):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class ada_mask(nn.Module):

    def __init__(self, input_channel):
        super(ada_mask, self).__init__()
        self.mask_head = nn.Conv2d(input_channel, 64, 3, 1, 1)
        self.mask_Res1 = ResBlock(64, 64, 3, 1, 1)
        self.mask_Res2 = ResBlock(64, 64, 3, 1, 1)
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.mask_Res1_1d = ResBlock(128, 128, 3, 1, 1)
        self.mask_Res1_2d = ResBlock(128, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.mask_Res2_1d = ResBlock(256, 256, 3, 1, 1)
        self.mask_Res2_2d = ResBlock(256, 256, 3, 1, 1)
        self.down3 = nn.Conv2d(256, 512, 3, 2, 1)
        self.mask_Res3_1d = ResBlock(512, 512, 3, 1, 1)
        self.mask_Res3_2d = ResBlock(512, 512, 3, 1, 1)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_Res3_1u = ResBlock(512, 256, 3, 1, 1)
        self.mask_Res3_2u = ResBlock(256, 256, 3, 1, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_Res2_1u = ResBlock(256, 128, 3, 1, 1)
        self.mask_Res2_2u = ResBlock(128, 128, 3, 1, 1)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_Res1_1u = ResBlock(128, 64, 3, 1, 1)
        self.mask_Res1_2u = ResBlock(64, 64, 3, 1, 1)
        self.mask_tail = nn.Conv2d(64, 26, 3, 1, 1)

    def forward(self, input):
        maskd0 = self.mask_Res2(self.mask_Res1(self.mask_head(input)))
        maskd1 = self.mask_Res1_2d(self.mask_Res1_1d(self.down1(maskd0)))
        maskd2 = self.mask_Res2_2d(self.mask_Res2_1d(self.down2(maskd1)))
        maskd3 = self.mask_Res3_2d(self.mask_Res3_1d(self.down3(maskd2)))
        masku2 = self.mask_Res3_2u(self.mask_Res3_1u(self.up3(maskd3))
            ) + maskd2
        masku1 = self.mask_Res2_2u(self.mask_Res2_1u(self.up2(masku2))
            ) + maskd1
        masku0 = self.mask_Res1_2u(self.mask_Res1_1u(self.up1(masku1))
            ) + maskd0
        mask = self.mask_tail(masku0)
        return mask


def get_inputs():
    return [torch.rand([4, 4, 8, 8])]


def get_init_inputs():
    return [[], {'input_channel': 4}]
