import torch
import torch.nn.functional as F


class CircPad(torch.nn.Module):

    def __init__(self, pad_size):
        super(CircPad, self).__init__()
        if type(pad_size) == tuple:
            self.padding = pad_size
        else:
            self.padding = tuple(pad_size for i in range(6))

    def forward(self, x):
        x = F.pad(x, self.padding, mode='circular')
        return x

    def __repr__(self):
        return f'{type(self).__name__}(pad_size={self.padding})'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pad_size': 4}]
