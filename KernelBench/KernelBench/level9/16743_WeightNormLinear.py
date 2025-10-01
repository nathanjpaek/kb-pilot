import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.onnx


class WeightNormLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(WeightNormLinear, self).__init__()
        self.lin = weight_norm(nn.Linear(in_features, out_features, bias))

    def forward(self, input):
        x = self.lin(input)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
