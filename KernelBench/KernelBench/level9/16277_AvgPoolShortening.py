from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class AvgPoolShortening(Module):
    """
    ### Average pool shortening

    This down-samples by a given factor with average pooling
    """

    def __init__(self, k: 'int'):
        """
        * `k` is the shortening factor
        """
        super().__init__()
        self.pool = nn.AvgPool1d(k, ceil_mode=True)

    def forward(self, x: 'torch.Tensor'):
        """
        * `x` is of shape `[seq_len, batch_size, d_model]`
        """
        return self.pool(x.permute(1, 2, 0)).permute(2, 0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'k': 4}]
