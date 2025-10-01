import torch
import torch.nn as nn


class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs
        ):
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
            dilation=dilation, **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
            dilation=dilation, **kwargs)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
