import torch
from torch import nn


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        super(_GlobalConvModule, self).__init__()
        self.pre_drop = nn.Dropout2d(p=0.1)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x = self.pre_drop(x)
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'kernel_size': [4, 4]}]
