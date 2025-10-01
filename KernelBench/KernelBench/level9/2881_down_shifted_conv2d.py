import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as wn


def down_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2] - 1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


class down_shifted_conv2d(nn.Module):

    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3),
        stride=(1, 1), shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size,
            stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), int((
            filter_size[1] - 1) / 2), filter_size[0] - 1, 0))
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)
        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 
                0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_filters_in': 4, 'num_filters_out': 4}]
