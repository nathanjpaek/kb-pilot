import torch
import torch.nn as nn


class _VariableWeightsAndBiases(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super(_VariableWeightsAndBiases, self).__init__()
        self.linear = nn.Linear(in_features, hidden_features)
        self.weights = nn.Linear(hidden_features, out_features)
        self.biases = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return self.weights(x), self.biases(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'hidden_features': 4, 'out_features': 4}]
