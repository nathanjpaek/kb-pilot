import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, in_channel=256, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel // ratio,
            kernel_size=1)
        self.conv_key = nn.Conv2d(in_channel, in_channel // ratio,
            kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        depth_q = self.conv_query(depth).view(bz, -1, h * w).permute(0, 2, 1)
        depth_k = self.conv_key(depth).view(bz, -1, h * w)
        mask = torch.bmm(depth_q, depth_k)
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(rgb).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1))
        feat = feat.view(bz, c, h, w)
        return feat


def get_inputs():
    return [torch.rand([4, 256, 64, 64]), torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
