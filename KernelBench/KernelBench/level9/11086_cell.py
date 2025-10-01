import math
import torch
import torch.nn as nn


class cell(nn.Module):

    def __init__(self, input_sz: 'int', hidden_sz: 'int', output_sz: 'int'):
        super().__init__()
        self.weights1 = nn.Parameter(torch.randn(input_sz, hidden_sz) /
            math.sqrt(input_sz), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(hidden_sz), requires_grad=True)
        self.weights2 = nn.Parameter(torch.randn(hidden_sz, output_sz) /
            math.sqrt(hidden_sz), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(output_sz), requires_grad=True)

    def forward(self, Input):
        a = nn.ReLU()(Input.clone() @ self.weights1 + self.bias1)
        a = nn.LogSigmoid()(a @ self.weights2 + self.bias2)
        return a


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_sz': 4, 'hidden_sz': 4, 'output_sz': 4}]
