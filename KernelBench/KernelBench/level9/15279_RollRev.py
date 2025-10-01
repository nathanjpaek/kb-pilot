import torch
from torch import nn


def roll(x, step, axis):
    shape = x.shape
    for i, s in enumerate(step):
        if s >= 0:
            x1 = x.narrow(axis[i], 0, s)
            x2 = x.narrow(axis[i], s, shape[axis[i]] - s)
        else:
            x2 = x.narrow(axis[i], shape[axis[i]] + s, -s)
            x1 = x.narrow(axis[i], 0, shape[axis[i]] + s)
        x = torch.cat([x2, x1], axis[i])
    return x


class RollRev(nn.Module):

    def __init__(self, step, axis):
        super(RollRev, self).__init__()
        if not isinstance(step, list):
            assert not isinstance(axis, list)
            step = [step]
            axis = [axis]
        assert len(step) == len(axis)
        self.step = step
        self.axis = axis

    def forward(self, x):
        return roll(x, self.step, self.axis)

    def reverse(self, x):
        return roll(x, [(-i) for i in self.step], self.axis)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'step': 4, 'axis': 4}]
