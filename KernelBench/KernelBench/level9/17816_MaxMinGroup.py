import torch
import torch.nn as nn
import torch.utils.data


def process_maxmin_groupsize(x, group_size, axis=-1):
    size = list(x.size())
    num_channels = size[axis]
    if num_channels % group_size:
        raise ValueError(
            'number of features({}) is not a multiple of group_size({})'.
            format(num_channels, num_channels))
    size[axis] = -1
    if axis == -1:
        size += [group_size]
    else:
        size.insert(axis + 1, group_size)
    return size


def maxout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]


class MaxMinGroup(nn.Module):

    def __init__(self, group_size, axis=-1):
        super(MaxMinGroup, self).__init__()
        self.group_size = group_size
        self.axis = axis

    def forward(self, x):
        maxes = maxout_by_group(x, self.group_size, self.axis)
        mins = minout_by_group(x, self.group_size, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'group_size: {}'.format(self.group_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'group_size': 4}]
