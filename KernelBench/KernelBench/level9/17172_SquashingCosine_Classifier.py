import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SquashingCosine_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001
        ):
        super(SquashingCosine_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims),
            requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = norm_x / (self.margin + norm_x) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dims': 4, 'out_dims': 4}]
