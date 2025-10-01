import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class GroupLinear(nn.Module):
    """
    Group Linear operator 
    """

    def __init__(self, in_planes, out_channels, groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes % groups == 0
        assert out_channels % groups == 0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim / self.groups)
        self.group_out_dim = int(self.out_dim / self.groups)
        self.group_weight = nn.Parameter(torch.zeros(self.groups, self.
            group_in_dim, self.group_out_dim))
        self.group_bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t, b, d = x.size()
        x = x.view(t, b, self.groups, int(d / self.groups))
        out = torch.einsum('tbgd,gdf->tbgf', (x, self.group_weight)).reshape(t,
            b, self.out_dim) + self.group_bias
        return out

    def extra_repr(self):
        s = '{in_dim}, {out_dim}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_channels': 4}]
