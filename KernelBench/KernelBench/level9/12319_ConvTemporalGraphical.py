import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        A_channels (int): Number of channels in the spatial adjacency matrix
        temporal_kernel_size (int): Size of temporal convolve kernel
        temporal_stride (int, optional): Stride of the temporal convolution. Default: 1
        temporal_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        temporal_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self, in_channels, out_channels, A_channels,
        temporal_kernel_size, temporal_stride=1, temporal_padding=0,
        temporal_dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * A_channels,
            kernel_size=(temporal_kernel_size, 1), padding=(
            temporal_padding, 0), stride=(temporal_stride, 1), dilation=(
            temporal_dilation, 1), bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, A.size(0), kc // A.size(0), t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'A_channels': 4,
        'temporal_kernel_size': 4}]
