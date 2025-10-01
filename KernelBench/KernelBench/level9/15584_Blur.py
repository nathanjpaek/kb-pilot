import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePad(nn.Module):

    def __init__(self, filter_size, pad_mode='constant', **kwargs):
        super(SamePad, self).__init__()
        self.pad_size = [int((filter_size - 1) / 2.0), int(math.ceil((
            filter_size - 1) / 2.0)), int((filter_size - 1) / 2.0), int(
            math.ceil((filter_size - 1) / 2.0))]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)
        return x

    def extra_repr(self):
        return 'pad_size=%s, pad_mode=%s' % (self.pad_size, self.pad_mode)


class Blur(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode='replicate', **
        kwargs):
        super(Blur, self).__init__()
        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)
        self.filter_proto = torch.tensor(sfilter, dtype=torch.float,
            requires_grad=False)
        self.filter = torch.tensordot(self.filter_proto, self.filter_proto,
            dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])
        return x

    def extra_repr(self):
        return 'pad=%s, filter_proto=%s' % (self.pad, self.filter_proto.
            tolist())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_filters': 4}]
