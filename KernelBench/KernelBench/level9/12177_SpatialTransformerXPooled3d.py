import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


def positive(weight, cache=None):
    weight.data *= weight.data.ge(0).float()
    return cache


class SpatialTransformerXPooled3d(nn.Module):

    def __init__(self, in_shape, outdims, pool_steps=1, positive=False,
        bias=True, init_range=0.2, grid_points=10, kernel_size=4, stride=4,
        grid=None, stop_grad=False):
        super().__init__()
        self._pool_steps = pool_steps
        self.in_shape = in_shape
        c, _t, _w, _h = in_shape
        self.outdims = outdims
        self.positive = positive
        self._grid_points = grid_points
        if grid is None:
            self.grid = Parameter(torch.Tensor(1, outdims, grid_points, 2))
        else:
            self.grid = grid
        self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1
            ), 1, outdims))
        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.avg = nn.AvgPool2d(kernel_size, stride=stride,
            count_include_pad=False)
        self.init_range = init_range
        self.initialize()
        self.stop_grad = stop_grad

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value
            ) - value == 0, 'new pool steps must be a non-negative integer'
        if value != self._pool_steps:
            None
            c, _t, _w, _h = self.in_shape
            outdims = self.outdims
            self._pool_steps = int(value)
            self.features = Parameter(torch.Tensor(1, c * (self._pool_steps +
                1), 1, outdims))
            self.features.data.fill_(1 / self.in_shape[0])

    def initialize(self, init_noise=0.001, grid=True):
        self.features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.bias.data.fill_(0)
        if grid:
            self.grid.data.uniform_(-self.init_range, self.init_range)

    def feature_l1(self, average=True, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if average:
            return self.features[..., subs_idx].abs().mean()
        else:
            return self.features[..., subs_idx].abs().sum()

    def dgrid_l2(self, average=True, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if average:
            return (self.grid[:, subs_idx, :-1, :] - self.grid[:, subs_idx,
                1:, :]).pow(2).mean()
        else:
            return (self.grid[:, subs_idx, :-1, :] - self.grid[:, subs_idx,
                1:, :]).pow(2).sum()

    def forward(self, x, shift=None, subs_idx=None):
        if self.stop_grad:
            x = x.detach()
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, t, w, h = x.size()
        m = self._pool_steps + 1
        if subs_idx is not None:
            feat = self.features[..., subs_idx].contiguous()
            outdims = feat.size(-1)
            feat = feat.view(1, m * c, outdims)
            grid = self.grid[:, subs_idx, ...]
        else:
            grid = self.grid
            feat = self.features.view(1, m * c, self.outdims)
            outdims = self.outdims
        if shift is None:
            grid = grid.expand(N * t, outdims, self._grid_points, 2)
        else:
            grid = grid.expand(N, outdims, self._grid_points, 2)
            grid = torch.stack([(grid + shift[:, i, :][:, None, None, :]) for
                i in range(t)], 1)
            grid = grid.contiguous().view(-1, outdims, self._grid_points, 2)
        z = x.contiguous().transpose(2, 1).contiguous().view(-1, c, w, h)
        pools = [F.grid_sample(z, grid, align_corners=True).mean(dim=3,
            keepdim=True)]
        for i in range(self._pool_steps):
            z = self.avg(z)
            pools.append(F.grid_sample(z, grid, align_corners=True).mean(
                dim=3, keepdim=True))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1) * feat).sum(1).view(N, t, outdims)
        if self.bias is not None:
            if subs_idx is None:
                y = y + self.bias
            else:
                y = y + self.bias[subs_idx]
        return y

    def __repr__(self):
        c, _, w, h = self.in_shape
        r = self.__class__.__name__ + ' (' + '{} x {} x {}'.format(c, w, h
            ) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'
        if self.stop_grad:
            r += ', stop_grad=True'
        r += '\n'
        for ch in self.children():
            r += '  -> ' + ch.__repr__() + '\n'
        return r


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_shape': [4, 4, 4, 4], 'outdims': 4}]
