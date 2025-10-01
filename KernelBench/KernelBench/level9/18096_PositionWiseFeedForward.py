from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFeedForward(nn.Module):

    def __init__(self, args):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.fc2 = nn.Linear(args.hidden_size * 4, args.hidden_size)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(hidden_size=4)}]
