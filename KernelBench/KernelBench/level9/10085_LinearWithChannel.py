import torch
import numpy as np
import torch.nn as nn


class LinearWithChannel(nn.Module):

    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        self.channel_size = channel_size
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
            input_size, output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size, 1,
            output_size))
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=np.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, observations):
        """
        observations = torch.tensor(batch_size, input_size)
        weight = torch.tensor(channel_size, input_size, output_size)
        bias = torch.tensor(channel_size, 1, output_size)
        :param observations:
        :return: torch.tensor(channel_size, batch_size, output_size)
        """
        observations = observations.repeat(self.channel_size, 1, 1)
        output = torch.bmm(observations, self.weight) + self.bias
        return output


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'channel_size': 4}]
