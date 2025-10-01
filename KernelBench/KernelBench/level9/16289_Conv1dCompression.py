from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class Conv1dCompression(Module):
    """
    ## 1D Convolution Compression $f_c$

    This is a simple wrapper around
    [`nn.Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
    with some tensor dimension permutations.
    """

    def __init__(self, compression_rate: 'int', d_model: 'int'):
        """
        * `compression_rate` $c$
        * `d_model` is the embedding size
        """
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=
            compression_rate, stride=compression_rate)

    def forward(self, mem: 'torch.Tensor'):
        """
        `mem` has shape `[seq_len, batch, d_model]`
        """
        mem = mem.permute(1, 2, 0)
        c_mem = self.conv(mem)
        return c_mem.permute(2, 0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'compression_rate': 4, 'd_model': 4}]
