import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x):
        output = self.lstm(x)
        return output[0]


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
