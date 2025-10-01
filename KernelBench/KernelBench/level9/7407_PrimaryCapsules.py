import torch
import torch.nn as nn


def squash(s, dim=-1):
    """
	"Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
	Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
	
	Args:
		s: 	Vector before activation
		dim:	Dimension along which to calculate the norm
	
	Returns:
		Squashed vector
	"""
    squared_norm = torch.sum(s ** 2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm
        ) + 1e-08)


class PrimaryCapsules(nn.Module):

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size=9,
        stride=2, padding=0):
        """
		Initialize the layer.

		Args:
			in_channels: 	Number of input channels.
			out_channels: 	Number of output channels.
			dim_caps:		Dimensionality, i.e. length, of the output capsule vector.
		
		"""
        super(PrimaryCapsules, self).__init__()
        self.dim_caps = dim_caps
        self._caps_channel = int(out_channels / dim_caps)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), self._caps_channel, out.size(2), out.
            size(3), self.dim_caps)
        out = out.view(out.size(0), -1, self.dim_caps)
        return squash(out)


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'dim_caps': 4}]
