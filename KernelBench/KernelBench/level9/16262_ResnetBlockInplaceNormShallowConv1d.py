import torch
import torch.nn as nn


class ResnetBlockInplaceNormShallowConv1d(nn.Module):
    """ Fully connected ResNet Block imeplemented with group convolutions and weight/spectral normalizations.

    Args:
        size_in (int): input dimension
        groups (int): number of groups for group convolutions
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, groups, norm_method='weight_norm', size_out
        =None, size_h=None, dropout_prob=0.0, leaky=False):
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
        fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups, bias=False)
        if norm_method == 'weight_norm':
            self.fc_0 = nn.utils.weight_norm(fc_0)
        elif norm_method == 'spectral_norm':
            self.fc_0 = nn.utils.spectral_norm(fc_0)
        else:
            raise ValueError('Normalization method {} not supported.'.
                format(norm_method))
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
            dx = self.fc_0(self.dropout(self.actvn(x)))
        else:
            dx = self.fc_0(self.actvn(x))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'size_in': 4, 'groups': 1}]
