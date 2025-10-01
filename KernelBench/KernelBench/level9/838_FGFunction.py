import torch
from torch import nn
from torch.nn import functional as F


class FGFunction(nn.Module):
    """Module used for F and G
	
	Archi :
	conv -> BN -> ReLu -> conv -> BN -> ReLu
	"""

    def __init__(self, channels):
        super(FGFunction, self).__init__()
        self.gn1 = nn.GroupNorm(1, channels, eps=0.001)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, channels, eps=0.001)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.gn1(F.leaky_relu(self.conv1(x), inplace=True))
        x = self.gn2(F.leaky_relu(self.conv2(x), inplace=True))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
