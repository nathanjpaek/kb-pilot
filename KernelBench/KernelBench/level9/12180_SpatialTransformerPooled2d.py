import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


def positive(weight, cache=None):
    weight.data *= weight.data.ge(0).float()
    return cache


class SpatialTransformerPooled2d(nn.Module):

    def __init__(self, in_shape, outdims, pool_steps=1, positive=False,
        bias=True, pool_kern=2, init_range=0.1):
        super().__init__()
        self._pool_steps = pool_steps
        self.in_shape = in_shape
        c, _w, _h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1
            ), 1, outdims))
        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.pool_kern = pool_kern
        self.avg = nn.AvgPool2d((pool_kern, pool_kern), stride=pool_kern,
            count_include_pad=False)
        self.init_range = init_range
        self.initialize()

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value
            ) - value == 0, 'new pool steps must be a non-negative integer'
        if value != self._pool_steps:
            None
            c, _w, _h = self.in_shape
            self._pool_steps = int(value)
            self.features = Parameter(torch.Tensor(1, c * (self._pool_steps +
                1), 1, self.outdims))
            self.features.data.fill_(1 / self.in_shape[0])

    def initialize(self):
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def group_sparsity(self, group_size):
        f = self.features.size(1)
        n = f // group_size
        ret = 0
        for chunk in range(0, f, group_size):
            ret = ret + (self.features[:, chunk:chunk + group_size, ...].
                pow(2).mean(1) + 1e-12).sqrt().mean() / n
        return ret

    def forward(self, x, shift=None, out_idx=None):
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, _w, _h = x.size()
        m = self.pool_steps + 1
        feat = self.features.view(1, m * c, self.outdims)
        if out_idx is None:
            grid = self.grid
            bias = self.bias
            outdims = self.outdims
        else:
            feat = feat[:, :, out_idx]
            grid = self.grid[:, out_idx]
            if self.bias is not None:
                bias = self.bias[out_idx]
            outdims = len(out_idx)
        if shift is None:
            grid = grid.expand(N, outdims, 1, 2)
        else:
            grid = grid.expand(N, outdims, 1, 2) + shift[:, None, None, :]
        pools = [F.grid_sample(x, grid, align_corners=True)]
        for _ in range(self.pool_steps):
            x = self.avg(x)
            pools.append(F.grid_sample(x, grid, align_corners=True))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)
        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + ' (' + '{} x {} x {}'.format(c, w, h
            ) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'
        r += ' and pooling for {} steps\n'.format(self.pool_steps)
        for ch in self.children():
            r += '  -> ' + ch.__repr__() + '\n'
        return r


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_shape': [4, 4, 4], 'outdims': 4}]
