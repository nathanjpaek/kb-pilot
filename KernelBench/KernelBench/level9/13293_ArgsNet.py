import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgsNet(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ArgsNet, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 50)
        self.fc2 = nn.Linear(50, self.input_size)

    def forward(self, input, hidden):
        new_hidden = self.gru(input, hidden)
        out = F.relu(self.fc1(new_hidden))
        out = self.fc2(out)
        return out, new_hidden


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
