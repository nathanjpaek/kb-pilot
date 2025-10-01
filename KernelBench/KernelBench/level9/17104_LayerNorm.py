import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization class. Normalization is done on the last dimension

    Args:
        input_size: size of input sample

    Inputs:
        a Tensor with shape (batch, length, input_size) or (batch, input_size)

    Outputs:
        a Tensor with shape (batch, length, input_size) or (batch, input_size)
    """

    def __init__(self, input_size, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(input_size))
        self.b = nn.Parameter(torch.zeros(input_size))

    def forward(self, input):
        mu = input.mean(-1).unsqueeze(-1)
        sigma = input.std(-1).unsqueeze(-1)
        output = (input - mu) / (sigma + self.eps)
        output = output * self.a.expand_as(output) + self.b.expand_as(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
