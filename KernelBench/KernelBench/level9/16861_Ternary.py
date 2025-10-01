import torch
from torch import nn


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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
