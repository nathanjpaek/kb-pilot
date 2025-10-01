import torch
import torch.nn as nn


class BasicLinearNet(nn.Module):

    def __init__(self, in_features, hidden_nodes, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, input):
        x = torch.tanh(self.linear1(input))
        return torch.tanh(self.linear2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'hidden_nodes': 4, 'out_features': 4}]
