import torch
import torch.nn.functional as F
import torch.nn as nn


class Dueling_Critic(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear1(x))
        y1 = self.linear2(x1)
        y2 = self.linear3(x2)
        x3 = y1 + y2 - y2.mean(dim=1, keepdim=True)
        return x3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}]
