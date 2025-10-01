import torch
import torch.nn as nn


class GatedConv2d(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\\phi(f(I))*\\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, bias=True, activation=torch.nn.ELU(1.0, inplace=True),
        dropout=0, gate_type='regular_conv'):
        super(GatedConv2d, self).__init__()
        self.stride = stride
        padding = dilation * (kernel_size - 1) // 2
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, bias=bias)
        if gate_type == 'regular_conv':
            self.mask_conv2d = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation, bias=bias)
        elif gate_type == 'single_channel':
            self.mask_conv2d = nn.Conv2d(in_channels=in_channels,
                out_channels=1, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias)
        elif gate_type == 'pixel_wise':
            self.mask_conv2d = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=1, stride=stride,
                padding=0, dilation=dilation, bias=bias)
        elif gate_type == 'depth_separable':
            self.mask_conv2d = nn.Sequential(nn.Conv2d(in_channels=
                in_channels, out_channels=in_channels, kernel_size=
                kernel_size, stride=stride, padding=padding, dilation=
                dilation, bias=bias, groups=in_channels), nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, padding=0, bias=bias))
        self.sigmoid = nn.Sigmoid()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.sigmoid(mask)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
