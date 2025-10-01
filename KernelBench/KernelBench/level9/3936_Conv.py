import torch
from torch import nn


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, equal_lr=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        norm_const = (in_channels * kernel_size ** 2) ** -0.5
        scale_init = 1 if equal_lr else norm_const
        self.scale_forward = norm_const if equal_lr else 1
        self.weight = nn.Parameter(scale_init * torch.randn(out_channels,
            in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, y_s=None, demod=False):
        weight = self.scale_forward * self.weight
        bias = self.bias
        groups = 1
        batch_size = x.size(0)
        if y_s is not None:
            weight = y_s.view(y_s.size(0), 1, y_s.size(1), 1, 1
                ) * weight.unsqueeze(0)
            if demod:
                x_s = ((weight ** 2).sum(dim=(2, 3, 4)) + 1e-08) ** 0.5
                weight = weight / x_s.view(*x_s.size(), 1, 1, 1)
            weight = weight.view(-1, *weight.size()[2:])
            bias = bias.expand(batch_size, -1).reshape(-1)
            groups = batch_size
            x = x.reshape(1, -1, *x.size()[2:])
        x = nn.functional.conv2d(x, weight, bias=bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=groups)
        return x.view(batch_size, -1, *x.size()[2:])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
