import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as F
import torch.utils.data.distributed


def get_same_padding(x: 'int', k: 'int', s: 'int', d: 'int'):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k, s, d=(1, 1), value=0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw,
        k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - 
            pad_h // 2], value=value)
    return x


class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channel, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=False, eps=1e-05):
        super().__init__(in_channel, out_channels, kernel_size, stride=
            stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True,
            unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return weight

    def forward(self, x):
        x = pad_same(x, self.get_weight().shape[-2:], self.stride, self.
            dilation)
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, (0, 0
            ), self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channels': 4, 'kernel_size': 4}]
