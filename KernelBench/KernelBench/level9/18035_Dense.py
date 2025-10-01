import math
import torch
import torch.nn as nn
from string import ascii_lowercase
import torch.optim


class Dense(nn.Module):

    def __init__(self, input_features, output_features=None):
        super(Dense, self).__init__()
        self.input_features = input_features
        self.output_features = (input_features if output_features is None else
            output_features)
        self.weight = nn.Parameter(torch.Tensor(input_features, self.
            output_features), requires_grad=True)
        self.weight.data.normal_(0, math.sqrt(2.0 / input_features))
        self.register_parameter('bias', None)

    def forward(self, x):
        return self.dense(x)

    def dense(self, inputs):
        eqn = 'ay{0},yz->az{0}'.format(ascii_lowercase[1:3])
        return torch.einsum(eqn, inputs, self.weight)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_features': 4}]
