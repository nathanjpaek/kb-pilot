import torch


class SeparableConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0,
            1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
