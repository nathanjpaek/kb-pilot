import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNGridworld(nn.Module):
    """
    Deep neural network with represents an agent.
    """

    def __init__(self, input_size, num_actions):
        super(DQNGridworld, self).__init__()
        self.linear1 = nn.Linear(input_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return self.head(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_actions': 4}]
