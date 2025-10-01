import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(torch.cat((x, -x), dim=1))


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


class GatedConv(nn.Module):
    """Gated Convolution Block

	Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).

	Args:
		num_channels (int): Number of channels in hidden activations.
		drop_prob (float): Dropout probability.
		aux_channels (int): Number of channels in optional auxiliary input.
	"""

    def __init__(self, num_channels, drop_prob=0.0, aux_channels=None):
        super(GatedConv, self).__init__()
        self.nlin = concat_elu
        self.conv = WNConv2d(2 * num_channels, num_channels, kernel_size=3,
            padding=1)
        self.drop = nn.Dropout2d(drop_prob)
        self.gate = WNConv2d(2 * num_channels, 2 * num_channels,
            kernel_size=1, padding=0)
        if aux_channels is not None:
            self.aux_conv = WNConv2d(2 * aux_channels, num_channels,
                kernel_size=1, padding=0)
        else:
            self.aux_conv = None

    def forward(self, x, aux=None):
        x = self.nlin(x)
        x = self.conv(x)
        if aux is not None:
            aux = self.nlin(aux)
            x = x + self.aux_conv(aux)
        x = self.nlin(x)
        x = self.drop(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
