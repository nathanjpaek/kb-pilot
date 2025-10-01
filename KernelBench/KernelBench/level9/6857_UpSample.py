import torch
import torch.nn as nn


class UpSample(nn.Module):

    def __init__(self, feat_in, feat_out, out_shape=None, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(feat_in, feat_out, kernel_size=(3, 3), stride
            =1, padding=1)
        self.out_shape, self.scale = out_shape, scale

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, size=self.out_shape,
            scale_factor=self.scale, mode='bilinear', align_corners=True))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feat_in': 4, 'feat_out': 4}]
