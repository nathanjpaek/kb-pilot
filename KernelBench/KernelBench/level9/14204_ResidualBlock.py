import torch
import torch.utils.data
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()
        self.fc_0 = nn.Conv1d(in_channels, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, out_channels, 1)
        self.activation = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        x_short = self.shortcut(x)
        x = self.fc_0(x)
        x = self.fc_1(self.activation(x))
        x = self.activation(x + x_short)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'hidden_dim': 4}]
