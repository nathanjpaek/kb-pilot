import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.policy = nn.Linear(100, output_size)
        self.value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.policy(x)
        value = self.value(x)
        return action, value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
