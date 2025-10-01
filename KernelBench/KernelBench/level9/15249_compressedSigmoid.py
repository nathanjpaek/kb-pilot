import torch
import torch.nn as nn
import torch._utils


class compressedSigmoid(nn.Module):

    def __init__(self, para=2.0, bias=0.2):
        super(compressedSigmoid, self).__init__()
        self.para = para
        self.bias = bias

    def forward(self, x):
        output = 1.0 / (self.para + torch.exp(-x)) + self.bias
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
