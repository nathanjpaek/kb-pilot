import torch
import torch.nn as nn


class NN(nn.Module):

    def __init__(self, input, hidden, output):
        super(NN, self).__init__()
        self.lin1 = nn.Linear(input, hidden)
        self.lin2 = nn.Linear(hidden, output)

    def forward(self, X):
        out = torch.sigmoid(self.lin1(X))
        out = torch.sigmoid(self.lin2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input': 4, 'hidden': 4, 'output': 4}]
