import torch
import torch.nn as nn


class IRevInjectivePad(nn.Module):
    """
    i-RevNet channel zero padding block.

    Parameters:
    ----------
    padding : int
        Size of the padding.
    """

    def __init__(self, padding):
        super(IRevInjectivePad, self).__init__()
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding=(0, 0, 0, padding))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.padding, :, :]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'padding': 4}]
