from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class SpatialDepthWisePerHeadConvolution(Module):
    """
    ## Spatial Depth Wise Per Head Convolution
    """

    def __init__(self, heads: 'int', d_k: 'int', kernel_size: 'int'=3):
        """
        * `heads` is the number of heads
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=d_k * heads, out_channels=d_k *
            heads, kernel_size=(kernel_size,), padding=(kernel_size - 1,),
            groups=d_k * heads)

    def forward(self, x: 'torch.Tensor'):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """
        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size, heads * d_k, seq_len)
        x = self.conv(x)
        x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'd_k': 4}]
