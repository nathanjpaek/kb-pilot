import torch


class TensorConstantLinear(torch.nn.Module):

    def __init__(self, weight=1, bias=0):
        self.weight = weight
        self.bias = bias
        super().__init__()

    def forward(self, input):
        return self.weight * input + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
