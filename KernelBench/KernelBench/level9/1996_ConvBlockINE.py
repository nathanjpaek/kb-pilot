import torch
from torch import nn
from torch.nn import init as init


class ConvBlockINE(nn.Module):

    def __init__(self, in_ch, out_ch, act='relu', ksize=3):
        super().__init__()
        padding = (ksize - 1) // 2
        if act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        else:
            self.act = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=
            padding, padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=ksize, padding=
            padding, padding_mode='circular')
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x, g=None, b=None):
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.norm1(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)
        out = self.norm2(x1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
