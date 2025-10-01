import torch


def setup_conv(in_channels, out_channels, kernel_size, bias, padding_mode,
    stride=1, Conv=torch.nn.Conv2d):
    return Conv(in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=
        stride, bias=bias)


class BilinearConvLayer(torch.nn.Module):

    def __init__(self, input_channels, output_channels, bilin_channels=None,
        padding_mode='zeros', Conv=torch.nn.Conv2d, nonlinearity=torch.nn.
        Identity(), norm=torch.nn.Identity(), kernel_size=3):
        super(BilinearConvLayer, self).__init__()
        bilin_channels = (output_channels if bilin_channels is None else
            bilin_channels)
        self.chgrp1 = max(0, output_channels - bilin_channels)
        self.chgrp2 = bilin_channels
        self.layer1 = setup_conv(in_channels=input_channels, out_channels=
            self.chgrp1 + 2 * self.chgrp2, kernel_size=kernel_size, bias=
            True, padding_mode=padding_mode, stride=1, Conv=Conv)
        self.norm = norm
        self.nonlinearity = nonlinearity

    def forward(self, x):
        y = self.nonlinearity(self.norm(self.layer1(x)))
        mid = self.chgrp1 + self.chgrp2
        y1, y2, y3 = y[:, :self.chgrp1], y[:, self.chgrp1:mid], y[:, mid:]
        z = y2 * y3
        out = torch.cat((y1, z), dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
