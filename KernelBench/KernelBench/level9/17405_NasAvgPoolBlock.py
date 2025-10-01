import torch
import torch.utils.data
import torch.nn as nn


class NasAvgPoolBlock(nn.Module):
    """
    NASNet specific 3x3 Average pooling layer with extra padding.

    Parameters:
    ----------
    extra_padding : bool, default False
        Whether to use extra padding.
    """

    def __init__(self, extra_padding=False):
        super(NasAvgPoolBlock, self).__init__()
        self.extra_padding = extra_padding
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
            count_include_pad=False)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
            x = x[:, :, 1:, 1:].contiguous()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
