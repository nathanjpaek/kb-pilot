import torch
import typing
import torch.multiprocessing
from torch import nn
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import torch.distributed


class FFN(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int',
        filter_channels: 'int', kernel_size: 'int', p_dropout: 'float'=0.0,
        activation: 'typing.Optional[str]'=None, causal: 'bool'=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == 'gelu':
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        x = F.pad(x, (pad_l, pad_r, 0, 0, 0, 0))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        x = F.pad(x, (pad_l, pad_r, 0, 0, 0, 0))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'filter_channels': 4,
        'kernel_size': 4}]
