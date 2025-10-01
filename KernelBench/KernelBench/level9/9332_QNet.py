import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        actions_value = self.out(x)
        return actions_value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
