import torch
from torch import nn


class _Linear(nn.Module):

    def __init__(self, input_dim=20, output_dim=10):
        super(_Linear, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.logprob = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Classify c"""
        x = self.fc1(x.view(-1, self.input_dim))
        return self.logprob(x)


def get_inputs():
    return [torch.rand([4, 20])]


def get_init_inputs():
    return [[], {}]
