import torch
import torch.nn as nn


class piNetwork(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, action_size):
        super(piNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, action_size)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.softmax(self.l3(x), dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size1': 4, 'hidden_size2': 4,
        'action_size': 4}]
