import torch
from torch import nn
from torch.nn import functional as F


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class PositionWiseFeedForwardNetworks(nn.Module):

    def __init__(self, input_size, output_size, d_ff):
        super().__init__()
        self.W_1 = Linear(input_size, d_ff, bias=True)
        self.W_2 = Linear(d_ff, output_size, bias=True)
        nn.init.constant_(self.W_1.bias, 0.0)
        nn.init.constant_(self.W_2.bias, 0.0)

    def forward(self, input):
        outputs = F.relu(self.W_1(input), inplace=True)
        return self.W_2(outputs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'd_ff': 4}]
