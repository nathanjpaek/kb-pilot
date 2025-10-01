import torch
import torch.nn as nn


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class GCN(nn.Module):

    def __init__(self, d_in, n_in, d_out=None, n_out=None, dropout=0.1):
        super().__init__()
        if d_out is None:
            d_out = d_in
        if n_out is None:
            n_out = n_in
        self.conv_n = nn.Conv1d(n_in, n_out, 1)
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = MemoryEfficientSwish()

    def forward(self, x):
        """
        :param x: [b, nin, din]
        :return: [b, nout, dout]
        """
        x = self.conv_n(x)
        x = self.dropout(self.linear(x))
        return self.activation(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'n_in': 4}]
