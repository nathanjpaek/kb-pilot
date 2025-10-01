import torch
import torch as t
import torch.nn as nn


class Squareplus(nn.Module):

    def __init__(self, a=2):
        super().__init__()
        self.a = a

    def forward(self, x):
        """The 'squareplus' activation function: has very similar properties to
        softplus, but is far cheaper computationally.
            - squareplus(0) = 1 (softplus(0) = ln 2)
            - gradient diminishes more slowly for negative inputs.
            - ReLU = (x + sqrt(x^2))/2
            - 'squareplus' becomes smoother with higher 'a'
        """
        return (x + t.sqrt(t.square(x) + self.a * self.a)) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
