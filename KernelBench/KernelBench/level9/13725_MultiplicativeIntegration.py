import torch
import torch.nn as nn
from typing import List


class MultiplicativeIntegration(nn.Module):

    def __init__(self, inputs_sizes: 'List[int]', output_sizes: 'List[int]',
        bias: 'bool', bias_start: 'float'=0.0, alpha_start: 'float'=1.0,
        beta_start: 'float'=1.0):
        super().__init__()
        self.inputs_sizes = inputs_sizes
        self.output_sizes = output_sizes
        total_output_size = sum(output_sizes)
        total_input_size = sum(inputs_sizes)
        self.bias_start = bias_start
        self.alpha_start = alpha_start
        self.beta_start = beta_start
        self.weights = nn.Parameter(torch.empty(total_input_size,
            total_output_size))
        self.alphas = nn.Parameter(torch.empty([total_output_size]))
        self.betas = nn.Parameter(torch.empty([2 * total_output_size]))
        self.biases = nn.Parameter(torch.empty([total_output_size])
            ) if bias else None
        self.reset_parameters()

    def forward(self, input0, input1):
        w1, w2 = torch.split(self.weights, self.inputs_sizes, dim=0)
        b1, b2 = torch.split(self.betas, sum(self.output_sizes), dim=0)
        wx1, wx2 = input0 @ w1, input1 @ w2
        res = self.alphas * wx1 * wx2 + b1 * wx1 + b2 * wx2
        if self.biases is not None:
            res += self.biases
        return res

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=1.0)
        nn.init.constant_(self.alphas, self.alpha_start)
        nn.init.constant_(self.betas, self.beta_start)
        if self.biases is not None:
            nn.init.constant_(self.biases, self.bias_start)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputs_sizes': [4, 4], 'output_sizes': [4, 4], 'bias': 4}]
