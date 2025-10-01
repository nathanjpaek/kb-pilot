from torch.nn import Module
import torch
from typing import *
import torch.utils.data
import torch.nn as nn
import torch.onnx.operators
import torch.optim


class HirarchicalAttention(Module):
    """
    ref: Hierarchical Attention Networks for Document Classiﬁcation
    """

    def __init__(self, hidden_size: 'int'):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(hidden_size, hidden_size)
        self.u_w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = (input * a_it).sum(dim=1)
        return s_i


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
