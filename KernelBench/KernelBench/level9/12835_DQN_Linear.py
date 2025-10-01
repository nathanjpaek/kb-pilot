import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_Linear(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN_Linear, self).__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 64)
        self.head = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.head(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
