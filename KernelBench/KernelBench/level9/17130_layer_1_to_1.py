import torch
import numpy as np
import torch.nn as nn


def contractions_1_to_1(inputs, dim, normalization='inf', normalization_val=1.0
    ):
    sum_all = torch.sum(inputs, dim=2).unsqueeze(dim=2)
    op1 = inputs
    op2 = torch.cat([sum_all for d in range(dim)], dim=2)
    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
    return [op1, op2]


class layer_1_to_1(nn.Module):
    """
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m tensor
    """

    def __init__(self, input_depth, output_depth, normalization='inf',
        normalization_val=1.0, device='cpu'):
        super().__init__()
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device
        self.basis_dimension = 2
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self
            .output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.
            input_depth + self.output_depth), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1))

    def forward(self, inputs):
        m = inputs.size(2)
        ops_out = contractions_1_to_1(inputs, m, normalization=self.
            normalization)
        ops_out = torch.stack(ops_out, dim=2)
        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)
        output = output + self.bias
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_depth': 1, 'output_depth': 1}]
