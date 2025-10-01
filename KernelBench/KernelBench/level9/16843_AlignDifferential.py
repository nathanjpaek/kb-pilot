import torch
from torch import nn


class AlignDifferential(nn.Module):

    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        padded_states = torch.cat([states[:, 0:1] * 2 - states[:, 1:2],
            states, states[:, -1:] * 2 - states[:, -2:-1]], dim=1)
        return (padded_states[:, 2:] - padded_states[:, :-2]) / 2

    def show(self, name='AlignDifferential', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = AlignDifferential()' % (name,))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
