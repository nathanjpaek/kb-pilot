import torch
from torch import nn
import torch.utils.data


class Highway(nn.Module):

    def __init__(self, input_dim, dropout):
        super(Highway, self).__init__()
        self.input_linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.gate_linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_):
        input_ = self.dropout(input_)
        output = self.relu(self.input_linear(input_))
        gate = self.sigmoid(self.gate_linear(input_))
        output = input_ * gate + output * (1.0 - gate)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'dropout': 0.5}]
