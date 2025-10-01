import torch
from torch import nn


def process_group_size(x, group_size, axis=-1):
    size = list(x.size())
    num_channels = size[axis]
    if num_channels % group_size:
        raise ValueError(
            'number of features({}) is not a multiple of group_size({})'.
            format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [group_size]
    else:
        size.insert(axis + 1, group_size)
    return size


def group_sort(x, group_size, axis=-1):
    size = process_group_size(x, group_size, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))
    return sorted_x


class GroupSort(nn.Module):

    def __init__(self, group_size, axis=-1):
        super(GroupSort, self).__init__()
        self.group_size = group_size
        self.axis = axis

    def forward(self, x):
        group_sorted = group_sort(x, self.group_size, self.axis)
        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'group_size': 4}]
