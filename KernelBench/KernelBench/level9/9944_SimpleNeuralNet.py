import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNeuralNet(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        return F.log_softmax(self.linear2(x), dim=-1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_hidden': 4, 'n_out': 4}]
