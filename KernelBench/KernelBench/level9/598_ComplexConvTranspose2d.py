import torch
import torch.nn as nn


class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1),
        stride=(1, 1), padding=(0, 0), output_padding=(0, 0), causal=False,
        complex_axis=1, groups=1):
        """
            in_channels: real+imag
            out_channels: real+imag
        """
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.
            out_channels, kernel_size, self.stride, padding=self.padding,
            output_padding=output_padding, groups=self.groups)
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.
            out_channels, kernel_size, self.stride, padding=self.padding,
            output_padding=output_padding, groups=self.groups)
        self.complex_axis = complex_axis
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)
            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)
        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
