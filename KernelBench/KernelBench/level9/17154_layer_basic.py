import torch
import numpy as np
import torch.nn as nn


class layer_basic(nn.Module):
    """
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    """

    def __init__(self, input_depth, output_depth, normalization='inf',
        normalization_val=1.0, device='cpu'):
        super().__init__()
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device
        self.basis_dimension = 4
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self
            .output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.
            input_depth + self.output_depth), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1))

    def forward(self, inputs):
        m = inputs.size(3)
        float_dim = np.float32(m)
        ops_out = []
        ops_out.append(inputs)
        sum_of_cols = torch.sum(inputs, dim=2) / float_dim
        ops_out.append(torch.cat([sum_of_cols.unsqueeze(dim=2) for i in
            range(m)], dim=2))
        sum_of_rows = torch.sum(inputs, dim=3) / float_dim
        ops_out.append(torch.cat([sum_of_rows.unsqueeze(dim=3) for i in
            range(m)], dim=3))
        sum_all = torch.sum(sum_of_rows, dim=2) / float_dim ** 2
        out = torch.cat([sum_all.unsqueeze(dim=2) for i in range(m)], dim=2)
        ops_out.append(torch.cat([out.unsqueeze(dim=3) for i in range(m)],
            dim=3))
        ops_out = torch.stack(ops_out, dim=2)
        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)
        output = output + self.bias
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_depth': 1, 'output_depth': 1}]
