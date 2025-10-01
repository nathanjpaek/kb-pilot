import torch
import torch.nn as nn


class ChannelMixer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=None):
        super(ChannelMixer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x + input
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
