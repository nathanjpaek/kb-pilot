from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class SpatialDepthWiseSharedConvolution(Module):
    """
    ## Spatial Depth Wise Shared Convolution

    We share the same kernel across all channels.
    """

    def __init__(self, kernel_size: 'int'=3):
        """
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(
            kernel_size,), padding=(kernel_size - 1,))

    def forward(self, x: 'torch.Tensor'):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """
        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size * heads * d_k, 1, seq_len)
        x = self.conv(x)
        x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
