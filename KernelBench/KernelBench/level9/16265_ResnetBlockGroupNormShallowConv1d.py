import torch
import torch.nn as nn


class GroupNorm1d(nn.Module):
    """ Group normalization that does per-point group normalization.

    Args:
        groups (int): number of groups
        f_dim (int): feature dimension, mush be divisible by groups
    """

    def __init__(self, groups, f_dim, eps=1e-05, affine=True):
        super().__init__()
        self.groups = groups
        self.f_dim = f_dim
        self.affine = affine
        self.eps = eps
        assert f_dim % groups == 0
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, f_dim, 1))
            self.beta = nn.Parameter(torch.zeros(1, f_dim, 1))

    def forward(self, x):
        batch_size, D, T = x.size()
        net = x.view(batch_size, self.groups, D // self.groups, T)
        means = net.mean(2, keepdim=True)
        variances = net.var(2, keepdim=True)
        net = (net - means) / (variances + self.eps).sqrt()
        net = net.view(batch_size, D, T)
        if self.affine:
            return net * self.gamma + self.beta
        else:
            return net


class ResnetBlockGroupNormShallowConv1d(nn.Module):
    """ Fully connected ResNet Block imeplemented with group convolutions and group normalizations.

    Args:
        size_in (int): input dimension
        groups (int): number of groups for group convolutions
        gn_groups (int): number of groups for group normalizations
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, groups, gn_groups=4, size_out=None, size_h=
        None, dropout_prob=0.0, leaky=False):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob, inplace=True)
        else:
            self.dropout = None
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.gn_0 = GroupNorm1d(groups * gn_groups, size_in)
        self.fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups, bias=False)
        if not leaky:
            self.actvn = nn.ReLU()
        else:
            self.actvn = nn.LeakyReLU(0.1)
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False,
                groups=groups)

    def forward(self, x):
        if self.dropout is not None:
            dx = self.fc_0(self.dropout(self.actvn(self.gn_0(x))))
        else:
            dx = self.fc_0(self.actvn(self.gn_0(x)))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'size_in': 4, 'groups': 1}]
