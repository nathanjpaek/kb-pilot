import torch
import torch.nn as nn


class FullyConnected2(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(FullyConnected2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        self.linear_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_layer_1 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):
        out = self.lrelu(self.linear_layer(input))
        return self.linear_layer_1(out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'output_size': 4}]
