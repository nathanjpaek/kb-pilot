import torch
from torch import nn


class FeedForwardNeuralNetModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetModel, self).__init__()
        self.linearA = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.linearB = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linearA(x)
        out = self.sigmoid(out)
        out = self.linearB(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
