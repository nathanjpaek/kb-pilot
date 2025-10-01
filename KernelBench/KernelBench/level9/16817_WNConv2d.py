import torch
import torch.nn as nn
import torch.utils.data


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.
	Args:
		in_channels (int): Number of channels in the input.
		out_channels (int): Number of channels in the output.
		kernel_size (int): Side length of each convolutional kernel.
		padding (int): Padding to add on edges of input.
		bias (bool): Use bias in the convolution operation.
	"""

    def __init__(self, in_channels, out_channels, kernel_size, padding,
        bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels,
            out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'padding': 4}]
