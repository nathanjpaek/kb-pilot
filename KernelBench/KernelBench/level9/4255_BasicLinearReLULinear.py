import torch
import torch.nn as nn


class BasicLinearReLULinear(nn.Module):

    def __init__(self, in_features, out_features=5, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(out_features, 1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
