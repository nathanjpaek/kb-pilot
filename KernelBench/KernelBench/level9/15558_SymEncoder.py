import torch
from torch import nn
import torch.utils.data


class SymEncoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(symmetry_size, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'symmetry_size': 4, 'hidden_size': 4}]
