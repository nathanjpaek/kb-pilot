import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.distributions


class Critic(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Critic, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return torch.clamp(x, 0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_outputs': 4}]
