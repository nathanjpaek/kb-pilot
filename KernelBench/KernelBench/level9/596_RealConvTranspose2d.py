import torch
import torch.nn as nn


class RealConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1),
        stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
        """
            in_channels: real+imag
            out_channels: real+imag
        """
        super(RealConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.conv = nn.ConvTranspose2d(self.in_channels, self.out_channels,
            kernel_size, self.stride, padding=self.padding, output_padding=
            output_padding, groups=self.groups)
        nn.init.normal_(self.conv.weight.data, std=0.05)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, inputs):
        out = self.conv(inputs)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
