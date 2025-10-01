import torch
import torch.nn as nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class ComplexConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias, **kwargs)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}]
