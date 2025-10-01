import torch
from torch import nn


class MaxPoolTrinary(nn.Module):

    def __init__(self):
        super().__init__()

    def new_length(self, length):
        return length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        assert states.size(1) >= 3
        side_length = (states.size(1) + 1) // 3
        return torch.cat([torch.max(states[:, :side_length], dim=1, keepdim
            =True)[0], torch.max(states[:, side_length:-side_length], dim=1,
            keepdim=True)[0], torch.max(states[:, -side_length:], dim=1,
            keepdim=True)[0]], dim=1)

    def show(self, name='MaxPoolTrinary', indent=0, log=print, **kwargs):
        log(' ' * indent + '- %s(x) = MaxPoolTrinary()' % (name,))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
