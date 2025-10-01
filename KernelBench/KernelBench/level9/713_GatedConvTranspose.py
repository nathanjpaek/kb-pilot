import torch
import torch.nn as nn
import torch.utils.data


class GatedConvTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding, groups=groups)
        self.layer_g = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding, groups=groups)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
