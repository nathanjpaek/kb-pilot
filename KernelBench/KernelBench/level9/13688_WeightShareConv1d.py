import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional
import torch.jit
import torch.nn.functional as F
import torch.utils.data
import torch.nn.utils


class VariationalHidDropout(nn.Module):

    def __init__(self, dropout=0.0):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every time step and every layer of TrellisNet
        :param dropout: The dropout rate (0 means no dropout is applied)
        :param temporal: Whether the dropout mask is the same across the temporal dimension (or only the depth dimension)
        """
        super(VariationalHidDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def reset_mask(self, x):
        dropout = self.dropout
        m = torch.zeros_like(x[:, :, :1]).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, 'You need to reset mask before using VariationalHidDropout'
        mask = self.mask.expand_as(x)
        return mask * x


class WeightShareConv1d(nn.Module):

    def __init__(self, n_hid, n_out, kernel_size, dropouth=0.0):
        """
        The weight-tied 1D convolution used in TrellisNet.
        :param n_hid: The dim of hidden input
        :param n_out: The dim of the pre-activation (i.e. convolutional) output
        :param kernel_size: The size of the convolutional kernel
        :param dropouth: Hidden-to-hidden dropout
        """
        super(WeightShareConv1d, self).__init__()
        self.kernel_size = kernel_size
        conv = nn.Conv1d(n_hid, n_out, kernel_size)
        self.weight = conv.weight
        self.bias = conv.bias
        self.init_weights()
        self.dict = dict()
        self.drop = VariationalHidDropout(dropout=dropouth)

    def init_weights(self):
        bound = 0.01
        self.weight.data.normal_(0, bound)
        self.bias.data.normal_(0, bound)

    def copy(self, func):
        self.weight.data = func.weight.data.clone().detach()
        self.bias.data = func.bias.data.clone().detach()
        self.drop.mask = func.drop.mask.clone().detach()

    def forward(self, x, dilation=1, hid=None):
        k = self.kernel_size
        padding = (k - 1) * dilation
        x = F.pad(x, (padding, 0))
        if hid is not None:
            x[:, :, :padding] = hid.repeat(1, 1, padding)
        res = F.conv1d(self.drop(x), self.weight, self.bias, dilation=dilation)
        return res


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_hid': 4, 'n_out': 4, 'kernel_size': 4}]
