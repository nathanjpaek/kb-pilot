import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, output_size)

    def forward(self, input):
        hidden = F.leaky_relu(self.fc1(input))
        hidden = F.leaky_relu(self.fc2(hidden))
        output = F.leaky_relu(self.fc3(hidden))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_1_size': 4, 'hidden_2_size': 4,
        'output_size': 4}]
