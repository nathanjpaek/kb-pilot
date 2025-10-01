import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterOutput(nn.Module):

    def __init__(self, in_features, out_features, low=-1, high=1):
        super(ParameterOutput, self).__init__()
        self.low = low
        self.high = high
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.softmax(self.linear(x))
        return (self.high - self.low) * x + self.low


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
