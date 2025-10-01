from _paritybench_helpers import _mock_config
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BiasAdd(nn.Module):

    def __init__(self, channels, opts, act='linear', alpha=None, gain=None,
        lrmul=1):
        """
            BiasAdd
        """
        super(BiasAdd, self).__init__()
        self.opts = opts
        self.bias = torch.nn.Parameter(torch.zeros(channels, 1, 1) * lrmul)
        self.act = act
        self.alpha = alpha if alpha is not None else 0.2
        self.gain = gain if gain is not None else 1.0

    def forward(self, x):
        x += self.bias
        if self.act == 'linear':
            pass
        elif self.act == 'lrelu':
            x = F.leaky_relu(x, self.alpha, inplace=True)
            x = x * np.sqrt(2)
        if self.gain != 1:
            x = x * self.gain
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'opts': _mock_config()}]
