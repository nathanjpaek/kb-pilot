import torch
from torch import nn


class MixPad2d(nn.Module):
    """Mixed padding modes for H and W dimensions

    Args:
        padding (tuple): the size of the padding for x and y, ie (pad_x, pad_y)
        modes (tuple): the padding modes for x and y, the values of each can be
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``['replicate', 'circular']``

    """
    __constants__ = ['modes', 'padding']

    def __init__(self, padding=[1, 1], modes=['replicate', 'circular']):
        super(MixPad2d, self).__init__()
        assert len(padding) == 2
        self.padding = padding
        self.modes = modes

    def forward(self, x):
        x = nn.functional.pad(x, (0, 0, self.padding[1], self.padding[1]),
            self.modes[1])
        x = nn.functional.pad(x, (self.padding[0], self.padding[0], 0, 0),
            self.modes[0])
        return x

    def extra_repr(self):
        repr_ = (
            'Mixed Padding: \t x axis: mode: {}, padding: {},\n\t y axis mode: {}, padding: {}'
            .format(self.modes[0], self.padding[0], self.modes[1], self.
            padding[1]))
        return repr_


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
