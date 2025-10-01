import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class Ternary(nn.Module):
    """
    Ternarize the input activations to -1, 0, 1.
    """

    def __init__(self, left=-0.25, right=0.25):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, input):
        input = input.clone()
        left_index = input.lt(self.left)
        right_index = input.ge(self.right)
        input[left_index] = -1
        input[right_index] = 1
        input[~(left_index | right_index)] = 0
        return input

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class TernaryLinear(nn.Module):

    def __init__(self, in_features, out_features, ternarize_left=-0.25,
        ternarize_right=0.25, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.ternarize = Ternary(ternarize_left, ternarize_right)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init.normal_(self.weight.data, 0, 1)

    def forward(self, input):
        return F.linear(input, self.ternarize(self.weight), self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
