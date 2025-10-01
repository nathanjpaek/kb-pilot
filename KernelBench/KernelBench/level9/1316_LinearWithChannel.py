import math
import torch
import torch.nn as nn


class LinearWithChannel(nn.Module):

    def __init__(self, channel_size, input_size, output_size):
        super(LinearWithChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
            output_size, input_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size, output_size))
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        assert x.ndim >= 2, 'Requires (..., channel, features) shape.'
        x = x.unsqueeze(-1)
        result = torch.matmul(self.weight, x).squeeze(-1) + self.bias
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel_size': 4, 'input_size': 4, 'output_size': 4}]
