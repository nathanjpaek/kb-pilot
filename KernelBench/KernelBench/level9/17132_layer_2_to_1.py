import torch
import numpy as np
import torch.nn as nn


def contractions_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0
    ):
    diag_part = torch.diagonal(inputs, dim1=2, dim2=3)
    sum_diag_part = torch.sum(diag_part, dim=2).unsqueeze(dim=2)
    sum_of_rows = torch.sum(inputs, dim=3)
    sum_of_cols = torch.sum(inputs, dim=2)
    sum_all = torch.sum(inputs, dim=(2, 3))
    op1 = diag_part
    op2 = torch.cat([sum_diag_part for d in range(dim)], dim=2)
    op3 = sum_of_rows
    op4 = sum_of_cols
    op5 = torch.cat([sum_all.unsqueeze(dim=2) for d in range(dim)], dim=2)
    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op3 = op3 / dim
            op4 = op4 / dim
            op5 = op5 / dim ** 2
    return [op1, op2, op3, op4, op5]


class layer_2_to_1(nn.Module):
    """
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
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
        self.basis_dimension = 5
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self
            .output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.
            input_depth + self.output_depth), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1))

    def forward(self, inputs):
        m = inputs.size(3)
        ops_out = contractions_2_to_1(inputs, m, normalization=self.
            normalization)
        ops_out = torch.stack(ops_out, dim=2)
        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)
        output = output + self.bias
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_depth': 1, 'output_depth': 1}]
