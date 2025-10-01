import math
import torch
import torch.nn as nn


class FastBiliner(nn.Module):

    def __init__(self, in1_features, in2_features, out_features):
        super(FastBiliner, self).__init__()
        weight = torch.randn(out_features, in1_features, in2_features
            ) * math.sqrt(2 / (in1_features + in2_features))
        bias = torch.ones(out_features) * math.sqrt(2 / (in1_features +
            in2_features))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.out_features = out_features
        self.in1_features = in1_features
        self.in2_features = in2_features

    def forward(self, input1, input2):
        assert len(input1.size()) == len(input2.size())
        input_dims = len(input1.size())
        weight_size = [1] * (input_dims - 2) + list(self.weight.size())
        bias_size = [1] * (input_dims - 2) + [self.out_features] + [1, 1]
        self.weight.view(*weight_size)
        bias = self.bias.view(*bias_size)
        input1 = input1.unsqueeze(-3)
        input2 = input2.unsqueeze(-3).transpose(-2, -1)
        outputs = bias + torch.matmul(input1, torch.matmul(self.weight.
            unsqueeze(0), input2))
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}]
