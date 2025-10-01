import torch
import torch.utils.data
import torch.nn as nn
import torch.optim
import torch.backends.cudnn
import torch.nn.functional as F


class NRelu(nn.Module):
    """
    -max(-x,0)
    Parameters
    ----------
    Input shape: (N, C, W, H)
    Output shape: (N, C * W * H)
    """

    def __init__(self, inplace):
        super(NRelu, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return -F.relu(-x, inplace=self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplace': 4}]
