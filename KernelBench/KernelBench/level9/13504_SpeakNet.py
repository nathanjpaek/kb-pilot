import math
import torch
import torch.nn as nn


def xavier_init(module):
    """Xavier initializer for module parameters."""
    for parameter in module.parameters():
        if len(parameter.data.shape) == 1:
            parameter.data.fill_(0)
        else:
            fan_in = parameter.data.size(0)
            fan_out = parameter.data.size(1)
            parameter.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))


class SpeakNet(nn.Module):
    """Module for speaking a token based on current state. In ``forward``:
    Return a probability distribution of utterances of tokens.
    """

    def __init__(self, state_size, out_size):
        super().__init__()
        self.net = nn.Linear(state_size, out_size)
        self.softmax = nn.Softmax()
        xavier_init(self)

    def forward(self, state):
        out_distr = self.softmax(self.net(state))
        return out_distr


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'out_size': 4}]
