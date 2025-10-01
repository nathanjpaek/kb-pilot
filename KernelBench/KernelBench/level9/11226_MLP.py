import torch
from torch import nn
import torch.utils.data


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=None, dropout=0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size * 4
        self.w_1 = nn.Linear(input_size * 2, hidden_size)
        self.w_2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], -1)
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
