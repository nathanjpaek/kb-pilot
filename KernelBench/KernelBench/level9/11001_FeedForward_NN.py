import torch
import torch.nn as nn


class FeedForward_NN(nn.Module):

    def __init__(self, input_size, hidden_layer, output_size):
        super(FeedForward_NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_layer, output_size)

    def forward(self, X):
        out = self.layer1(X)
        out = self.relu(out)
        out = self.layer2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_layer': 1, 'output_size': 4}]
