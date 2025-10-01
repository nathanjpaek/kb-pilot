import torch
from torch import nn
from torch.nn import init as init


class ConvBlockLNEDense(nn.Module):

    def __init__(self, n_ch, act='relu', ksize=3):
        super().__init__()
        padding = (ksize - 1) // 2
        if act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        else:
            self.act = nn.ReLU(True)
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=ksize, padding=
            padding, padding_mode='circular')
        self.conv2 = nn.Conv2d(2 * n_ch, n_ch, kernel_size=ksize, padding=
            padding, padding_mode='circular')
        self.conv3 = nn.Conv2d(3 * n_ch, n_ch, kernel_size=ksize, padding=
            padding, padding_mode='circular')
        self.conv4 = nn.Conv2d(4 * n_ch, n_ch, kernel_size=ksize, padding=
            padding, padding_mode='circular')
        self.norm1 = nn.GroupNorm(1, n_ch, affine=True)
        self.norm2 = nn.GroupNorm(1, n_ch, affine=True)
        self.norm3 = nn.GroupNorm(1, n_ch, affine=True)

    def forward(self, x, g=None, b=None):
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.norm1(x1)
        x2 = torch.cat([x1, x], dim=1)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        x2 = self.norm2(x2)
        x3 = torch.cat([x2, x1, x], dim=1)
        x3 = self.conv3(x3)
        x3 = self.act(x3)
        x3 = self.norm3(x3)
        x4 = torch.cat([x3, x2, x1, x], dim=1)
        out = self.conv4(x4)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_ch': 4}]
