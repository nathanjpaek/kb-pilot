import torch
from torch import nn


class Differential(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def new_length(self, length):
        new_length = (length + self.padding * 2 - self.kernel_size + 1
            ) // self.stride
        return new_length

    def forward(self, states):
        """
        :param states: [batch, length, *]
        """
        _batch, length, _n_agents, _state_dim = states.size()
        padding = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        if padding != 0:
            if padding > 0:
                states = torch.cat([states[:, :1].repeat(1, padding, 1, 1),
                    states, states[:, -1:].repeat(1, padding, 1, 1)], dim=1)
            else:
                states = states[:, -padding:padding]
        new_length = (length + padding * 2 - kernel_size + 1) // stride
        differentials = states[:, 0:new_length * stride:stride] - states[:,
            kernel_size - 1:kernel_size - 1 + new_length * stride:stride]
        return differentials

    def show(self, name='Differential', indent=0, log=print, **kwargs):
        log(' ' * indent + 
            '- %s(x) = Differential(ks=%d, stride=%d, padding=%d)' % (name,
            self.kernel_size, self.stride, self.padding))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
