from torch.nn import Module
import math
import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GaussLinearStandardized(Module):

    def __init__(self, in_features, out_features, bias=True,
        raw_weight_variance=1.0, raw_bias_variance=1.0):
        super(GaussLinearStandardized, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight_variance = raw_weight_variance
        self.raw_bias_variance = raw_bias_variance
        self.epsilon_weight = Parameter(torch.Tensor(out_features, in_features)
            )
        if bias:
            self.epsilon_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('epsilon_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.epsilon_weight.data.normal_()
        if self.epsilon_bias is not None:
            self.epsilon_bias.data.normal_()

    def forward(self, input):
        stdv = 1.0 / math.sqrt(self.in_features)
        weight = self.epsilon_weight * stdv * math.sqrt(self.
            raw_weight_variance)
        if self.epsilon_bias is not None:
            bias = self.epsilon_bias * math.sqrt(self.raw_bias_variance)
        else:
            bias = None
        return F.linear(input, weight, bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
