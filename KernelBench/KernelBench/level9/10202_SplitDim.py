import torch
from torch import nn as nn
import torch.utils.data


class SplitDim(nn.Module):

    def __init__(self, nonlin_col=1, nonlin_type=torch.nn.functional.
        softplus, correction=True):
        super(SplitDim, self).__init__()
        self.nonlinearity = nonlin_type
        self.col = nonlin_col
        if correction:
            self.var = torch.nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('var', torch.ones(1, requires_grad=False) *
                -15.0)
        self.correction = correction

    def forward(self, input):
        transformed_output = self.nonlinearity(input[:, self.col])
        transformed_output = transformed_output + self.nonlinearity(self.var)
        stack_list = [input[:, :self.col], transformed_output.view(-1, 1)]
        if self.col + 1 < input.size(1):
            stack_list.append(input[:, self.col + 1:])
        output = torch.cat(stack_list, 1)
        return output


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
